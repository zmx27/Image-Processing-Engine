// Phase 8's error-injection pass (docs/PLAN.md), and Colab-only — every case needs a
// real NVIDIA GPU. Three claims, all about what happens when something goes WRONG:
//
//   * a genuine illegal access poisons the context, and the backend recovers from it;
//   * malformed protocol frames are answered without the GPU ever hearing about them;
//   * a driver fault under a live server costs the frames it hit and nothing else.
//
// WHY THIS IS NOT ALREADY COVERED. Phase 6's [recovery] cases call recreate_context()
// directly and feed the backend a job with a null input — neither produces a *sticky
// driver error*, which is the thing recovery exists for and the one state that cannot be
// reached by asking politely. Phase 4's [server] cases cover every malformed header, but
// against the CPU backend, so they say nothing about whether a protocol error can reach
// the GPU. This file closes both gaps, on real hardware.
//
// HOW A FAULT IS INJECTED WITHOUT BREAKING INVARIANT 1. Only the worker thread may make
// a driver call. Two situations here, handled differently:
//
//   * Backend-level cases construct the CudaBackend on the test's own thread, so the
//     context is current *here* and the injection is an ordinary same-thread launch.
//   * The server owns its backend on its worker thread, and reaching in from the test
//     thread would be exactly the violation the invariant exists to prevent. So the
//     server case injects through a test-only IBackend decorator instead: submit() is
//     already called on the worker thread (imgjit/backend/backend.h), which makes it the
//     one legal place to fault the context of a running server. No production code has a
//     test hook in it.
//
// The fault itself is a null-pointer store from a kernel. That is a genuine
// CUDA_ERROR_ILLEGAL_ADDRESS and it is permanent: every later call on that context
// returns it, which is precisely why the only way back is to destroy and rebuild.
//
// EXPECT NOISE ON STDERR FROM THESE CASES. The fault module belongs to the context the
// recovery then destroys, so unloading it later fails and RAII teardown reports it
// ("imgjit: ignoring failure during teardown: ... cuModuleUnload ...", see
// src/backend/cuda/cuda_check.h). That line is the design working as intended — teardown
// after a poisoned context reports and swallows rather than throwing — and not a symptom
// of a failing test. Catch2 reporting the case as passed is the thing to read.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "backend/cuda/cuda_backend.h"
#include "backend/cuda/cuda_raii.h"
#include "backend/cuda/nvrtc_compile.h"
#include "catch_amalgamated.hpp"
#include "imgjit/backend/cpu/ops.h"
#include "imgjit/core/image.h"
#include "imgjit/core/op_chain.h"
#include "imgjit/net/client.h"
#include "imgjit/net/protocol.h"
#include "imgjit/net/server.h"

using imgjit::CudaBackend;
using imgjit::Image;
using imgjit::OpChain;
using imgjit::net::Client;
using imgjit::net::ClientResponse;
using imgjit::net::RequestHeader;
using imgjit::net::Server;
using imgjit::net::ServerConfig;
using imgjit::net::Status;

namespace {

Image make_image(int width, int height, int channels, std::uint32_t seed) {
  Image image(width, height, channels);
  std::uint32_t state = seed | 1U;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        state = state * 1664525U + 1013904223U;
        image.at(x, y, c) = static_cast<std::uint8_t>(state >> 24U);
      }
    }
  }
  return image;
}

OpChain chain_for(std::string_view text) {
  const auto parsed = imgjit::parse_op_chain(text);
  REQUIRE(parsed.has_value());
  return *parsed;
}

int max_abs_difference(const Image& lhs, const Image& rhs) {
  if (lhs.width() != rhs.width() || lhs.height() != rhs.height() ||
      lhs.channels() != rhs.channels() || lhs.byte_count() != rhs.byte_count()) {
    return 256;
  }
  int worst = 0;
  for (std::size_t i = 0; i < lhs.byte_count(); ++i) {
    const int difference = static_cast<int>(lhs.data()[i]) - static_cast<int>(rhs.data()[i]);
    worst = std::max(worst, difference < 0 ? -difference : difference);
  }
  return worst;
}

Image oracle(const Image& input, std::string_view chain_text) {
  return imgjit::cpu::apply_chain(input, chain_for(chain_text));
}

// A store through a null pointer. Nothing about this kernel depends on the op set or on
// codegen — it exists only to make the driver angry, and it is the smallest thing that
// does so reliably: device address 0 is never mapped, so this is an illegal access on
// every device rather than a corruption that might go unnoticed.
constexpr const char* kFaultKernel = R"CUDA(
extern "C" __global__ void imgjit_force_fault(int* target) {
  target[threadIdx.x] = 0x0BADF00D;
}
)CUDA";

// Compiles the fault kernel for `arch`. Must be called on the thread that owns the
// context (NVRTC itself needs none, but the module load does).
imgjit::cuda::CudaModule build_fault_module(const std::string& arch, std::string& ptx_out) {
  ptx_out = imgjit::cuda::compile_to_ptx(kFaultKernel, "imgjit_fault.cu", arch);
  return imgjit::cuda::CudaModule(ptx_out.c_str());
}

// Launches the fault and waits for it. Returns the CUresult the synchronize reported,
// which the caller checks is actually an error — an injection that silently succeeded
// would leave every assertion after it passing for the wrong reason.
//
// Deliberately NOT wrapped in CU_CHECK: the whole point is to leave the error standing
// in the context rather than convert it into an exception here.
CUresult inject_illegal_access(const imgjit::cuda::CudaModule& module) {
  const CUfunction fault = module.get_function("imgjit_force_fault");
  CUdeviceptr target = 0;  // never mapped
  void* arguments[] = {&target};
  const CUresult launched =
      cuLaunchKernel(fault, 1, 1, 1, 32, 1, 1, 0, nullptr, arguments, nullptr);
  if (launched != CUDA_SUCCESS) {
    return launched;  // some drivers reject it at launch; either way the context is done
  }
  return cuCtxSynchronize();
}

ServerConfig fault_config() {
  ServerConfig config;
  config.port = 0;
  config.num_slots = 4;
  config.slots_per_connection = 2;
  config.max_payload_bytes = 256 * 1024;
  config.queue_capacity = 4;
  config.output_dir = std::string(IMGJIT_TEST_TMP_DIR) + "/phase8_faults_out";
  return config;
}

// TEST-ONLY. Forwards everything to a real CudaBackend, and when armed, poisons the
// context from inside submit() — which the server calls on its worker thread, making
// this the one place a running server's context can legally be faulted (see the file
// header). Nothing in src/ knows this exists.
class FaultInjectingBackend final : public imgjit::IBackend {
 public:
  FaultInjectingBackend() {
    // On the worker thread, before the server accepts anything: compile the fault kernel
    // now so that triggering it later is a bare launch and cannot fail for any reason
    // other than the fault itself.
    fault_module_.emplace(build_fault_module(inner_.context().compute_arch(), fault_ptx_));
  }

  std::byte* allocate_slots(std::size_t count, std::size_t bytes) override {
    return inner_.allocate_slots(count, bytes);
  }

  imgjit::JobHandle submit(const imgjit::FrameJob& job) override {
    if (armed_.exchange(false)) {
      last_injection_ = inject_illegal_access(*fault_module_);
      injections_.fetch_add(1);
    }
    return inner_.submit(job);
  }

  std::vector<imgjit::Completion> poll_completions() override { return inner_.poll_completions(); }

  imgjit::BackendStats stats() const override { return inner_.stats(); }

  // Called from the test thread; the worker picks it up on its next submit(). Only the
  // flag crosses threads, never a driver call.
  void arm() { armed_.store(true); }
  std::uint32_t injections() const { return injections_.load(); }
  CUresult last_injection() const { return last_injection_; }

 private:
  // Declaration order is load-bearing: the module must be unloaded before the context it
  // belongs to is destroyed, and destruction runs in reverse.
  CudaBackend inner_;
  std::string fault_ptx_;
  std::optional<imgjit::cuda::CudaModule> fault_module_;

  std::atomic<bool> armed_{false};
  std::atomic<std::uint32_t> injections_{0};
  CUresult last_injection_{CUDA_SUCCESS};
};

RequestHeader header_for(const Image& image, std::uint32_t seq_num, std::size_t chain_len) {
  RequestHeader header;
  header.flags = imgjit::net::kFlagEcho;
  header.seq_num = seq_num;
  header.width = static_cast<std::uint32_t>(image.width());
  header.height = static_cast<std::uint32_t>(image.height());
  header.channels = static_cast<std::uint8_t>(image.channels());
  header.payload_len = static_cast<std::uint32_t>(image.byte_count());
  header.chain_len = static_cast<std::uint16_t>(chain_len);
  return header;
}

std::vector<std::uint8_t> payload_of(const Image& image) {
  return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
}

}  // namespace

TEST_CASE("a forced illegal access is recovered from", "[faults]") {
  // The case Phase 6 could not write: a real sticky driver error, not a call to
  // recreate_context(). Everything the recovery path claims is asserted on the far side
  // of an actual CUDA_ERROR_ILLEGAL_ADDRESS.
  constexpr int kSize = 64;
  const std::size_t bytes = static_cast<std::size_t>(kSize) * kSize * 3;
  const Image input = make_image(kSize, kSize, 3, 0xfa17U);
  const OpChain chain = chain_for("grayscale,gaussian:1.4");
  const Image expected = imgjit::cpu::apply_chain(input, chain);

  CudaBackend backend(0, 2);
  std::byte* const slots_before = backend.allocate_slots(2, bytes);

  const auto run = [&](std::size_t slot) {
    std::memcpy(slots_before + slot * bytes, input.data(), input.byte_count());
    imgjit::FrameJob job;
    job.input = slots_before + slot * bytes;
    job.width = input.width();
    job.height = input.height();
    job.channels = input.channels();
    job.stride = input.stride();
    job.chain = chain;
    const imgjit::JobHandle handle = backend.submit(job);
    std::vector<imgjit::Completion> completions;
    while (completions.empty()) {
      std::vector<imgjit::Completion> batch = backend.poll_completions();
      completions.insert(completions.end(), std::make_move_iterator(batch.begin()),
                         std::make_move_iterator(batch.end()));
    }
    REQUIRE(completions.size() == 1);
    REQUIRE(completions.front().handle == handle);
    return completions.front();
  };

  // Healthy first, so the fault is the only thing that changes.
  const imgjit::Completion before = run(0);
  INFO("pre-fault error: " << before.error);
  REQUIRE(before.status == imgjit::JobStatus::kOk);
  REQUIRE(backend.compile_count() == 1);
  REQUIRE(backend.stats().context_recreations == 0);

  // The context is current on this thread, because this thread constructed the backend.
  std::string ptx;
  const imgjit::cuda::CudaModule fault_module =
      build_fault_module(backend.context().compute_arch(), ptx);
  const CUresult injected = inject_illegal_access(fault_module);
  // If this ever comes back CUDA_SUCCESS the injection did nothing and every assertion
  // below would pass vacuously, so it is the first thing checked.
  REQUIRE(injected != CUDA_SUCCESS);
  INFO("injected result " << static_cast<int>(injected));

  // The next frame dies on the poisoned context and takes the recovery path with it.
  const imgjit::Completion hit = run(1);
  CHECK(hit.status == imgjit::JobStatus::kError);
  CHECK_FALSE(hit.error.empty());
  CHECK(backend.stats().context_recreations == 1);

  // ...and the backend is usable again afterwards, on a rebuilt context with a flushed
  // cache. The compile counter climbing is the observable half of "the cache went with
  // the context" — a recreation that kept stale CUmodules would launch a function
  // belonging to a context that no longer exists.
  const imgjit::Completion after = run(0);
  INFO("post-recovery error: " << after.error);
  REQUIRE(after.status == imgjit::JobStatus::kOk);
  CHECK(max_abs_difference(expected, after.output) <= 1);
  CHECK(backend.compile_count() == 2);
  CHECK(backend.stats().context_recreations == 1);
}

TEST_CASE("malformed frames are answered without touching the GPU", "[faults]") {
  // Phase 4 proved each of these against the CPU backend. The claim here is narrower and
  // specific to this phase: a protocol error is rejected in the network layer and never
  // becomes a GPU event, so no amount of malformed traffic can provoke a context
  // recreation. The counter at the bottom is the whole point of the case.
  const Image input = make_image(32, 24, 3, 0xbadf00dU);
  Server server(fault_config(), [] {
    return std::unique_ptr<imgjit::IBackend>(std::make_unique<CudaBackend>());
  });
  server.start();

  // Drain-and-continue dispositions share one connection: after each rejection the next
  // frame must still be served correctly, which is what proves the payload was drained.
  {
    Client client("127.0.0.1", server.port());

    const std::string bad_chain = "not-an-op";
    client.send_raw(header_for(input, 1, bad_chain.size()), bad_chain, payload_of(input));
    CHECK(client.receive().status == Status::kBadChain);

    const std::string long_chain(imgjit::net::kMaxChainLen + 17, 'x');
    client.send_raw(header_for(input, 2, long_chain.size()), long_chain, payload_of(input));
    CHECK(client.receive().status == Status::kBadChain);

    RequestHeader lying = header_for(input, 3, 6);
    lying.width = input.width() + 1;
    client.send_raw(lying, "invert", payload_of(input));
    CHECK(client.receive().status == Status::kLengthMismatch);

    // Still in sync, and the GPU still serves this connection.
    client.send(input, "grayscale,sobel");
    const ClientResponse served = client.receive();
    REQUIRE(served.matched);
    INFO("status " << static_cast<int>(served.status));
    REQUIRE(served.status == Status::kOk);
    CHECK(max_abs_difference(oracle(input, "grayscale,sobel"), served.image) <= 1);
  }

  // Close dispositions, one connection each: the server hangs up after answering, and
  // must survive every one of them.
  {
    Client client("127.0.0.1", server.port());
    RequestHeader header = header_for(input, 1, 6);
    header.magic = 0x0BADF00D;
    client.send_raw(header, "invert", payload_of(input));
    CHECK(client.receive().status == Status::kBadMagic);
    CHECK_THROWS(client.receive());
  }
  {
    Client client("127.0.0.1", server.port());
    RequestHeader header = header_for(input, 1, 6);
    header.version = 99;
    client.send_raw(header, "invert", payload_of(input));
    const ClientResponse rejected = client.receive();
    CHECK(rejected.status == Status::kUnsupportedVersion);
    CHECK(rejected.version == imgjit::net::kVersion);
    CHECK_THROWS(client.receive());
  }
  {
    Client client("127.0.0.1", server.port());
    RequestHeader header = header_for(input, 1, 6);
    header.width = 8192;
    header.height = 8192;
    header.payload_len = 8192U * 8192U * 3U;  // far over the configured cap
    client.send_raw(header, "invert", {});    // and the bytes never arrive
    CHECK(client.receive().status == Status::kPayloadTooLarge);
    CHECK_THROWS(client.receive());
  }

  // A fresh connection after all of that is served normally: the server is alive and the
  // acceptor was never disturbed.
  {
    Client client("127.0.0.1", server.port());
    client.send(input, "invert");
    const ClientResponse served = client.receive();
    REQUIRE(served.status == Status::kOk);
    CHECK(max_abs_difference(oracle(input, "invert"), served.image) <= 1);
  }

  // backend_stats() is snapshotted by the worker on its way out and reads as zeroes
  // before that (imgjit/net/server.h), so the stop() is what makes the next line an
  // assertion rather than a tautology.
  server.stop();

  // THE ASSERTION THIS CASE EXISTS FOR. Six malformed frames, and the GPU never heard
  // about any of them.
  CHECK(server.backend_stats().context_recreations == 0);
  CHECK(server.frames_completed() == 2);  // the two good frames, and only those
}

TEST_CASE("a driver fault under load costs the frames it hit and nothing else", "[faults]") {
  // The end-to-end version of case 1: a real illegal access, injected on the worker
  // thread of a running server (see the file header on why it has to be), with clients
  // attached. docs/PROTOCOL.md's contract for status 6 is that the server stays alive —
  // this is that contract under an actual driver fault rather than a malformed job.
  const Image input = make_image(48, 32, 3, 0x5eedU);
  const std::string chain = "gaussian:1.4";
  const Image expected = oracle(input, chain);

  std::atomic<FaultInjectingBackend*> injector{nullptr};
  Server server(fault_config(), [&injector] {
    auto backend = std::make_unique<FaultInjectingBackend>();
    injector.store(backend.get());
    return std::unique_ptr<imgjit::IBackend>(std::move(backend));
  });
  server.start();
  REQUIRE(injector.load() != nullptr);

  Client client("127.0.0.1", server.port());

  // Healthy traffic first. (No stats assertion here: backend_stats() is snapshotted on
  // the worker's way out and reads as zeroes until stop(), so checking it mid-run would
  // assert nothing — every stats check in this case happens after the stop() below.)
  for (int i = 0; i < 3; ++i) {
    client.send(input, chain);
    const ClientResponse response = client.receive();
    REQUIRE(response.status == Status::kOk);
  }

  // Arm, then keep sending. The worker faults its own context on the next submit; the
  // frames caught by it come back as status 6, and the backend rebuilds underneath.
  injector.load()->arm();
  int failed = 0;
  int served_after = 0;
  int worst = 0;
  for (int i = 0; i < 8; ++i) {
    client.send(input, chain);
    const ClientResponse response = client.receive();
    REQUIRE(response.matched);
    if (response.status == Status::kInternalError) {
      ++failed;
    } else if (response.status == Status::kOk) {
      ++served_after;
      worst = std::max(worst, max_abs_difference(expected, response.image));
    }
  }

  // A connection opened after the fault is served normally too: recovery is not limited
  // to the connection that happened to be attached when it happened.
  Client fresh("127.0.0.1", server.port());
  fresh.send(input, chain);
  const ClientResponse late = fresh.receive();
  REQUIRE(late.matched);
  INFO("post-fault status " << static_cast<int>(late.status));
  REQUIRE(late.status == Status::kOk);
  CHECK(max_abs_difference(expected, late.image) <= 1);

  // Read the decorator's own counters BEFORE stopping: the worker owns the backend and
  // destroys it on the way out, so this pointer dangles once stop() returns.
  const std::uint32_t injections = injector.load()->injections();
  const CUresult last_injection = injector.load()->last_injection();
  server.stop();

  INFO("last injection result " << static_cast<int>(last_injection));
  // If the injection never happened, every assertion below would be about a fault that
  // was not injected — so it is checked first, and checked exactly.
  CHECK(injections == 1);
  CHECK(last_injection != CUDA_SUCCESS);
  // At least one frame paid for the fault.
  CHECK(failed >= 1);
  // ...and the connection kept working afterwards, on the rebuilt context.
  CHECK(served_after >= 1);
  CHECK(worst <= 1);
  CHECK(server.backend_stats().context_recreations >= 1);

  // The frame slots are the server's for the life of the process and reader threads
  // write into them concurrently, so a recreation that moved or freed them would corrupt
  // live connections instead of recovering. Nothing above would have worked if it had.
  CHECK(server.frames_failed() == static_cast<std::uint64_t>(failed));
}
