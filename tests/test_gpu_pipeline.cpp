// Phase 6's gate (docs/PLAN.md), and Colab-only — every case here needs a real NVIDIA
// GPU. Three claims, registered as three ctest cases so a failure names which broke:
//
//   [async]     submit() returns while the frame is still on the GPU, several frames
//               are genuinely resident at once, and every handle comes back exactly
//               once with its own chain's answer.
//   [stress]    sustained traffic through the real server, with output checksums, aimed
//               squarely at premature slot reuse (invariant 3).
//   [recovery]  context recreation as a unit: in-flight jobs die with status 6, the
//               kernel cache is flushed, the frame slots survive, work resumes.
//
// The stress case is the one that earns its runtime. Invariant 3 is the sharpest hazard
// in the design precisely because breaking it does not crash: a slot handed back before
// its DMA finished is overwritten by the next recv() mid-transfer, and the frame that
// comes out is merely WRONG. Nothing short of comparing outputs notices. So the
// comparison is not against the oracle alone — GPU and CPU differ by up to 1 LSB, which
// would mask exactly the corruption being hunted — but against the GPU's own earlier
// answer for the same bytes, which must be identical every single time.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "backend/cuda/cuda_backend.h"
#include "catch_amalgamated.hpp"
#include "imgjit/backend/cpu/ops.h"
#include "imgjit/core/image.h"
#include "imgjit/core/op_chain.h"
#include "imgjit/net/client.h"
#include "imgjit/net/server.h"

using imgjit::CudaBackend;
using imgjit::Image;
using imgjit::OpChain;
using imgjit::net::Client;
using imgjit::net::ClientResponse;
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
    return 256;  // any mismatch in shape fails the tolerance check outright
  }
  int worst = 0;
  for (std::size_t i = 0; i < lhs.byte_count(); ++i) {
    const int difference = static_cast<int>(lhs.data()[i]) - static_cast<int>(rhs.data()[i]);
    worst = std::max(worst, difference < 0 ? -difference : difference);
  }
  return worst;
}

// Mean absolute difference over every byte. This is what separates float rounding from
// corruption in the stress test: GPU/CPU disagreement on a stencil is a handful of
// knife-edge samples out of ~200k, so the mean is ~0.01, while a single frame written
// into a slot whose previous transfer had not finished is a contiguous wrong block that
// pulls the mean past 1 on its own.
double mean_abs_difference(const Image& lhs, const Image& rhs) {
  if (lhs.byte_count() != rhs.byte_count() || lhs.byte_count() == 0) {
    return 256.0;
  }
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < lhs.byte_count(); ++i) {
    const int difference = static_cast<int>(lhs.data()[i]) - static_cast<int>(rhs.data()[i]);
    total += static_cast<std::uint64_t>(difference < 0 ? -difference : difference);
  }
  return static_cast<double>(total) / static_cast<double>(lhs.byte_count());
}

// FNV-1a over the output bytes. Cheap enough to run on every one of hundreds of frames,
// and the only property asked of it is that a single corrupted byte changes it.
std::uint64_t checksum(const Image& image) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t i = 0; i < image.byte_count(); ++i) {
    hash ^= image.data()[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

// Submits `job` and blocks until its completion appears, collecting anything else that
// retires alongside it. Used where a test wants one frame's answer rather than the
// pipeline's behaviour.
std::vector<imgjit::Completion> drain(CudaBackend& backend, std::size_t expected) {
  std::vector<imgjit::Completion> collected;
  while (collected.size() < expected) {
    std::vector<imgjit::Completion> batch = backend.poll_completions();
    collected.insert(collected.end(), std::make_move_iterator(batch.begin()),
                     std::make_move_iterator(batch.end()));
  }
  return collected;
}

// A backend with its frame slots already allocated, which is the only state in which
// submit() will run anything (invariant 2 — there is no allocating path).
class Pipeline {
 public:
  Pipeline(std::size_t streams, std::size_t slot_count, std::size_t slot_bytes)
      : backend_(0, streams),
        base_(backend_.allocate_slots(slot_count, slot_bytes)),
        slot_count_(slot_count),
        slot_bytes_(slot_bytes) {}

  // Copies `image` into slot `index` and queues it. Returns immediately: from Phase 6
  // the frame is still on the GPU when this comes back.
  imgjit::JobHandle submit(std::size_t index, const Image& image, const OpChain& chain) {
    REQUIRE(index < slot_count_);
    REQUIRE(image.byte_count() <= slot_bytes_);
    std::byte* slot = base_ + index * slot_bytes_;
    std::memcpy(slot, image.data(), image.byte_count());

    imgjit::FrameJob job;
    job.input = slot;
    job.width = image.width();
    job.height = image.height();
    job.channels = image.channels();
    job.stride = image.stride();
    job.chain = chain;
    return backend_.submit(job);
  }

  CudaBackend& backend() { return backend_; }
  std::byte* base() const { return base_; }

 private:
  CudaBackend backend_;
  std::byte* base_;
  std::size_t slot_count_;
  std::size_t slot_bytes_;
};

ServerConfig stress_config() {
  ServerConfig config;
  config.port = 0;
  // Fewer slots than the traffic wants, on purpose: the pool recycles constantly, which
  // is the state a premature return corrupts. Two per connection means several readers
  // are parked on slot claim at any moment.
  config.num_slots = 4;
  config.slots_per_connection = 2;
  config.max_payload_bytes = 256 * 1024;
  config.queue_capacity = 8;
  config.output_dir = std::string(IMGJIT_TEST_TMP_DIR) + "/phase6_out";
  return config;
}

// What one stress client saw, checked on the main thread — Catch2's macros are not
// thread-safe.
struct StressResult {
  std::uint32_t frames_ok{0};
  // Worst max-abs and worst mean-abs difference against the CPU oracle, over the first
  // sighting of each variant. The pair is the backstop against a pipeline that corrupts
  // deterministically (which `drifted` alone would not catch): a stencil's float
  // divergence is a few LSB on a scatter of knife-edge samples, so `worst_difference`
  // stays small and `worst_mean` stays near zero, while a half-written frame is a
  // contiguous wrong block that blows past both.
  int worst_difference{0};
  double worst_mean{0.0};
  // variant index -> the checksum the GPU produced the FIRST time it saw those bytes.
  // Every later sighting must match it exactly.
  std::unordered_map<int, std::uint64_t> reference;
  int drifted{0};
  std::string error;
};

// One connection, pipelining a repeating cycle of distinct payloads through a fixed
// chain. Distinct payloads are what make slot reuse observable: if a slot were released
// while its transfer was still running, the next frame's recv() would overwrite it and
// the answer would be neither variant's.
void run_stress_client(std::uint16_t port, const std::vector<Image>& variants,
                       const std::vector<Image>& expected, const std::string& chain, int rounds,
                       int window, StressResult& result) {
  try {
    Client client("127.0.0.1", port);
    const int total = rounds * static_cast<int>(variants.size());
    int sent = 0;
    int received = 0;

    while (received < total) {
      while (sent < total && sent - received < window) {
        client.send(variants[static_cast<std::size_t>(sent) % variants.size()], chain);
        ++sent;
      }
      const ClientResponse response = client.receive();
      ++received;
      if (!response.matched) {
        result.error = "a response arrived for a seq_num that was never sent";
        return;
      }
      if (response.status != Status::kOk) {
        result.error = "status " + std::to_string(static_cast<int>(response.status));
        return;
      }

      // seq_num is 1-based and assigned in send order, so it names the variant even
      // though responses may arrive in any order — which under K streams they do.
      const int variant = static_cast<int>((response.seq_num - 1) % variants.size());
      const std::uint64_t seen = checksum(response.image);
      const auto known = result.reference.find(variant);
      if (known == result.reference.end()) {
        result.reference.emplace(variant, seen);
        const Image& want = expected[static_cast<std::size_t>(variant)];
        result.worst_difference =
            std::max(result.worst_difference, max_abs_difference(want, response.image));
        result.worst_mean = std::max(result.worst_mean, mean_abs_difference(want, response.image));
      } else if (known->second != seen) {
        ++result.drifted;
      }
      ++result.frames_ok;
    }
  } catch (const std::exception& error) {
    result.error = error.what();
  }
}

}  // namespace

TEST_CASE("submit returns with the frame still on the GPU and K run at once", "[async]") {
  // gaussian:4 is the heaviest kernel the closed op set can produce — radius 12, so a
  // 25x25 stencil per output sample — and 512x512x3 makes one frame milliseconds of GPU
  // work against microseconds of host work to queue it. That gap is what lets the test
  // assert real residency rather than hoping the scheduler cooperated.
  constexpr std::size_t kStreams = 4;
  constexpr int kSize = 512;
  const std::size_t bytes = static_cast<std::size_t>(kSize) * kSize * 3;

  Pipeline pipeline(kStreams, kStreams, bytes);
  const OpChain chain = chain_for("gaussian:4");
  // Warmed first, so the NVRTC compile on frame 1 is not what the other three overlap
  // with. Occupancy is a claim about transfers and launches, not about the compiler.
  pipeline.backend().prewarm(chain, 3);

  std::vector<Image> inputs;
  std::vector<imgjit::JobHandle> handles;
  for (std::size_t i = 0; i < kStreams; ++i) {
    inputs.push_back(make_image(kSize, kSize, 3, 0x51ed270bU + static_cast<std::uint32_t>(i)));
  }
  for (std::size_t i = 0; i < kStreams; ++i) {
    handles.push_back(pipeline.submit(i, inputs[i], chain));
  }

  const std::vector<imgjit::Completion> completions = drain(pipeline.backend(), kStreams);
  REQUIRE(completions.size() == kStreams);

  // Every handle back exactly once, each carrying its own frame's answer. Matching by
  // handle rather than by arrival order is the whole reason handles exist — with K
  // streams the Nth completion is not the Nth submission.
  std::unordered_map<imgjit::JobHandle, const imgjit::Completion*> by_handle;
  for (const imgjit::Completion& completion : completions) {
    by_handle.emplace(completion.handle, &completion);
  }
  REQUIRE(by_handle.size() == kStreams);
  for (std::size_t i = 0; i < kStreams; ++i) {
    const auto found = by_handle.find(handles[i]);
    REQUIRE(found != by_handle.end());
    INFO("backend error: " << found->second->error);
    REQUIRE(found->second->status == imgjit::JobStatus::kOk);
    CHECK(max_abs_difference(imgjit::cpu::apply_chain(inputs[i], chain),
                             found->second->output) <= 1);
  }

  // The measurement the phase gate turns on. A pipeline that serialized would report a
  // peak of 1 no matter how fast it ran, so this is the difference between "faster" and
  // "actually overlapped" (docs/PLAN.md Phase 6).
  const imgjit::BackendStats stats = pipeline.backend().stats();
  INFO("peak in flight " << stats.max_in_flight << ", mean " << stats.mean_in_flight);
  CHECK(stats.max_in_flight >= 2);
  CHECK(stats.frames_submitted == kStreams);
  CHECK(stats.context_recreations == 0);
}

TEST_CASE("more frames than streams stall rather than overrun", "[async]") {
  // Past K in flight there is nowhere to put a frame, so submit() waits for a stream to
  // retire. That wait is the backend's own backpressure and it is counted, because a
  // pipeline that silently ran a frame in a buffer another frame was still using would
  // look exactly like one that waited (invariant 3).
  constexpr std::size_t kStreams = 2;
  constexpr std::size_t kFrames = 12;
  constexpr int kSize = 96;
  const std::size_t bytes = static_cast<std::size_t>(kSize) * kSize * 4;

  Pipeline pipeline(kStreams, 1, bytes);
  const OpChain chain = chain_for("grayscale,gaussian:1.4,sobel");
  const Image input = make_image(kSize, kSize, 4, 0xfeedU);
  const Image expected = imgjit::cpu::apply_chain(input, chain);

  // One slot, reused immediately: legal only because the completion is what says the
  // transfer out of it finished. Writing the next frame into it before that would be
  // the invariant-3 violation, and here it would show as a wrong answer below.
  std::vector<imgjit::Completion> collected;
  for (std::size_t i = 0; i < kFrames; ++i) {
    pipeline.submit(0, input, chain);
    std::vector<imgjit::Completion> batch = pipeline.backend().poll_completions();
    collected.insert(collected.end(), std::make_move_iterator(batch.begin()),
                     std::make_move_iterator(batch.end()));
  }
  while (collected.size() < kFrames) {
    std::vector<imgjit::Completion> batch = pipeline.backend().poll_completions();
    collected.insert(collected.end(), std::make_move_iterator(batch.begin()),
                     std::make_move_iterator(batch.end()));
  }

  REQUIRE(collected.size() == kFrames);
  for (const imgjit::Completion& completion : collected) {
    INFO("backend error: " << completion.error);
    REQUIRE(completion.status == imgjit::JobStatus::kOk);
  }
  // Same bytes in, same kernel, 12 times through 2 stream slots that are reused ~5 times
  // each — so every output must be bit-identical to the first. This is the invariant-3
  // check here: a device buffer handed back before its event completed would give one
  // frame different bytes from the rest.
  for (const imgjit::Completion& completion : collected) {
    CHECK(completion.output == collected.front().output);
  }
  // Loose oracle backstop — sobel's float divergence can reach a few LSB (see the
  // reasoning in the [stress] case), so this only asserts the kernel is not wildly
  // wrong, not <=1.
  CHECK(max_abs_difference(expected, collected.front().output) <= 8);

  const imgjit::BackendStats stats = pipeline.backend().stats();
  CHECK(stats.frames_submitted == kFrames);
  CHECK(stats.max_in_flight <= kStreams);
  CHECK(stats.submit_stalls > 0);
  CHECK(pipeline.backend().compile_count() == 1);
}

TEST_CASE("sustained traffic produces no checksum drift", "[stress]") {
  // docs/PLAN.md Phase 6's stress gate. Four connections share four slots, so the pool
  // turns over continuously; each connection cycles eight distinct payloads so that a
  // slot released one frame early yields a checksum that is nobody's answer.
  constexpr int kWidth = 256;
  constexpr int kHeight = 256;
  constexpr int kVariants = 8;
  constexpr int kRounds = 25;  // 200 frames per connection, 800 in total
  constexpr int kWindow = 4;   // frames in flight per connection, above the slot cap

  struct Workload {
    std::string chain;
    int channels;
  };
  // No `threshold` after a stencil in this corpus, on purpose. That pairing is a
  // discontinuity — a sub-ULP GPU/CPU disagreement on a post-sobel sample that lands on
  // the threshold flips it 0<->255 — so the <=1 LSB oracle bound below genuinely does
  // not hold against the random inputs this test generates (phase3_gpu_oracle_diff
  // covers that fusion with a curated input instead). This test is about checksum
  // stability under slot recycling; the chains just need to span the fusion shapes:
  // prologue fusion + two stencils, a lone stencil with alpha passthrough, a stencil
  // with an epilogue, and a pointwise-only single-channel run.
  const std::vector<Workload> workloads{
      {"grayscale,gaussian:1.4,sobel", 3},
      {"gaussian:1.4", 4},
      {"sobel,invert", 3},
      {"brightness:0.2,grayscale", 1},
  };

  Server server(stress_config(), [] {
    return std::unique_ptr<imgjit::IBackend>(std::make_unique<CudaBackend>(0, 4));
  });
  server.start();

  std::vector<std::vector<Image>> variants(workloads.size());
  std::vector<std::vector<Image>> expected(workloads.size());
  std::vector<StressResult> results(workloads.size());
  for (std::size_t w = 0; w < workloads.size(); ++w) {
    const OpChain chain = chain_for(workloads[w].chain);
    for (int v = 0; v < kVariants; ++v) {
      variants[w].push_back(make_image(kWidth, kHeight, workloads[w].channels,
                                       0x2545f491U * static_cast<std::uint32_t>(w + 1) +
                                           static_cast<std::uint32_t>(v)));
      expected[w].push_back(imgjit::cpu::apply_chain(variants[w].back(), chain));
    }
  }

  std::vector<std::thread> threads;
  threads.reserve(workloads.size());
  for (std::size_t w = 0; w < workloads.size(); ++w) {
    threads.emplace_back([&, w] {
      run_stress_client(server.port(), variants[w], expected[w], workloads[w].chain, kRounds,
                        kWindow, results[w]);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  server.stop();

  const std::uint32_t per_client = static_cast<std::uint32_t>(kRounds) * kVariants;
  for (std::size_t w = 0; w < workloads.size(); ++w) {
    INFO("chain \"" << workloads[w].chain << "\" at " << workloads[w].channels << " channels");
    CHECK(results[w].error.empty());
    CHECK(results[w].frames_ok == per_client);
    // Zero, not "small". Two runs of one kernel over identical bytes on one GPU are
    // bit-identical, so any drift at all is a frame that read memory it did not own.
    // This is the actual invariant-3 gate.
    CHECK(results[w].drifted == 0);
    // And the answers are right in the first place, not merely stable — a pipeline that
    // corrupted every copy of a variant identically would pass the drift check alone.
    // Not <=1 LSB: `sobel` is a difference operator (|coefficients| sum to 8), so
    // GPU/CPU float reassociation amplifies to ~4 LSB on a scatter of samples
    // (docs/ARCHITECTURE.md decision 5), and over ~200k samples per frame that tail is
    // sampled every run. The mean is what makes this a corruption check rather than a
    // numerics check: rounding noise leaves it near zero, a contiguous wrong block from
    // a half-written slot does not.
    INFO("worst LSB " << results[w].worst_difference << ", mean " << results[w].worst_mean);
    CHECK(results[w].worst_difference <= 8);
    CHECK(results[w].worst_mean < 0.25);
  }

  CHECK(server.frames_completed() ==
        static_cast<std::uint64_t>(workloads.size()) * per_client);
  CHECK(server.frames_failed() == 0);

  const imgjit::BackendStats stats = server.backend_stats();
  INFO("peak in flight " << stats.max_in_flight << ", mean " << stats.mean_in_flight);
  CHECK(stats.max_in_flight >= 2);
  CHECK(stats.context_recreations == 0);
  // Backpressure engaged somewhere: either readers waited on the slot pool or the
  // backend waited on a stream. If neither happened the load never actually stressed
  // anything and the checks above proved less than they look like they did.
  CHECK((server.slot_waits() > 0 || stats.submit_stalls > 0));
}

TEST_CASE("context recreation kills in-flight work and resumes", "[recovery]") {
  // docs/PLAN.md Phase 6. An illegal access poisons a context permanently, so recovery
  // is destroying it together with everything whose lifetime it owns — the kernel
  // cache, the device pool and the in-flight table — and rebuilding the set. This
  // asserts each half of that: what is lost, and what comes back.
  constexpr std::size_t kStreams = 4;
  constexpr int kSize = 512;
  const std::size_t bytes = static_cast<std::size_t>(kSize) * kSize * 3;

  Pipeline pipeline(kStreams, kStreams, bytes);
  const OpChain chain = chain_for("gaussian:4");
  const Image input = make_image(kSize, kSize, 3, 0xa5a5U);
  const Image expected = imgjit::cpu::apply_chain(input, chain);
  std::byte* const slots_before = pipeline.base();

  for (std::size_t i = 0; i < kStreams; ++i) {
    pipeline.submit(i, input, chain);
  }
  REQUIRE(pipeline.backend().compile_count() == 1);

  // Four 25x25-stencil frames over a quarter-megapixel each are milliseconds of GPU
  // work; queueing them was microseconds. They are still resident, which is what makes
  // the assertion below about in-flight jobs rather than about finished ones.
  pipeline.backend().recreate_context("forced by the recovery test");

  const std::vector<imgjit::Completion> lost = drain(pipeline.backend(), kStreams);
  REQUIRE(lost.size() == kStreams);
  for (const imgjit::Completion& completion : lost) {
    // JobStatus::kError is what the server turns into protocol status 6, which is the
    // documented cost of a driver fault: the frames it hit, and nothing else.
    CHECK(completion.status == imgjit::JobStatus::kError);
    CHECK_FALSE(completion.error.empty());
  }

  // The frame slots survived. They must: the server's slot pool holds this pointer for
  // the life of the process and reader threads write into it concurrently, so a
  // recreation that moved or freed it would corrupt live connections rather than
  // recover from anything (src/backend/cuda/cuda_raii.h, RegisteredHostBuffer).
  CHECK(pipeline.base() == slots_before);

  // The cache went with the context, so the same chain is a miss again — which is the
  // observable half of "flush the kernel cache". A recreation that left stale CUmodules
  // behind would launch a function belonging to a destroyed context.
  pipeline.submit(0, input, chain);
  const std::vector<imgjit::Completion> resumed = drain(pipeline.backend(), 1);
  REQUIRE(resumed.size() == 1);
  INFO("backend error: " << resumed.front().error);
  REQUIRE(resumed.front().status == imgjit::JobStatus::kOk);
  CHECK(max_abs_difference(expected, resumed.front().output) <= 1);
  CHECK(pipeline.backend().compile_count() == 2);
  CHECK(pipeline.backend().stats().context_recreations == 1);
}

TEST_CASE("a chain that cannot compile does not take the context down", "[recovery]") {
  // The distinction NvrtcError exists for (src/backend/cuda/cuda_check.h). A compile
  // failure never touched a context, so recovering from one by tearing the GPU down
  // would turn a single bad job into every other client's failed frame. There is no way
  // to make the generated source invalid through the parser, so this checks the
  // neighbouring property the same code path guarantees: a malformed job is one error
  // completion, the context is untouched, and the next frame is served normally.
  Pipeline pipeline(2, 1, 64 * 64 * 3);
  const OpChain chain = chain_for("grayscale,sobel");
  const Image input = make_image(64, 64, 3, 0x1234U);

  imgjit::FrameJob bad;
  bad.input = nullptr;
  bad.chain = chain;
  const imgjit::JobHandle bad_handle = pipeline.backend().submit(bad);

  const std::vector<imgjit::Completion> failed = drain(pipeline.backend(), 1);
  REQUIRE(failed.size() == 1);
  CHECK(failed.front().handle == bad_handle);
  CHECK(failed.front().status == imgjit::JobStatus::kError);

  pipeline.submit(0, input, chain);
  const std::vector<imgjit::Completion> served = drain(pipeline.backend(), 1);
  REQUIRE(served.size() == 1);
  INFO("backend error: " << served.front().error);
  CHECK(served.front().status == imgjit::JobStatus::kOk);
  // <=8, not <=1: this chain has a sobel, whose float divergence from the oracle can
  // reach a few LSB on noise (see the [stress] case). The point here is that the frame
  // was served correctly at all, not its exact numerics.
  CHECK(max_abs_difference(imgjit::cpu::apply_chain(input, chain), served.front().output) <= 8);
  // The point of the case: nothing was rebuilt over a bad job.
  CHECK(pipeline.backend().stats().context_recreations == 0);
}
