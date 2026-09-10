// Phase 5's gate (docs/PLAN.md): the GPU worker behind the real server, driven by
// concurrent loopback clients asking for different chains, diffed against the CPU
// oracle. Colab-only — every case here needs a real NVIDIA GPU.
//
// This is the phase where the two tracks merge, so it is worth being precise about what
// it adds over the suites that already exist. Phase 4's [server] tests prove the
// protocol and the threading against the CPU backend; Phase 3's [oracle] tests prove
// the generated kernels against the same oracle single-threaded. Neither covers the
// seam between them, and the seam is where Phase 5's specific hazards live:
//
//   * the pinned slot pool is allocated by the worker and recv()'d into by connection
//     threads, so a frame's bytes cross a thread boundary without a copy;
//   * one CUcontext and one kernel cache serve every connection, so chains from
//     different clients interleave through a single compile-and-launch path;
//   * a channel count that differs per connection is a different kernel identity
//     (CLAUDE.md invariant 4), so a shared-cache mistake shows up as one client getting
//     another client's kernel — wrong pixels, not a crash.
//
// Catch2 assertion macros are not thread-safe, so every client thread records what it
// saw and the checks happen on the main thread after the join.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "backend/cuda/cuda_backend.h"
#include "catch_amalgamated.hpp"
#include "imgjit/backend/cpu/ops.h"
#include "imgjit/core/image.h"
#include "imgjit/core/op_chain.h"
#include "imgjit/net/client.h"
#include "imgjit/net/server.h"

using imgjit::Image;
using imgjit::net::Client;
using imgjit::net::ClientResponse;
using imgjit::net::Server;
using imgjit::net::ServerConfig;
using imgjit::net::Status;

namespace {

Image make_image(int width, int height, int channels, std::uint32_t seed) {
  Image image(width, height, channels);
  std::uint32_t state = seed;
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

int max_abs_difference(const Image& lhs, const Image& rhs) {
  if (lhs.width() != rhs.width() || lhs.height() != rhs.height() ||
      lhs.channels() != rhs.channels() || lhs.byte_count() != rhs.byte_count()) {
    return 256;  // any mismatch in shape fails the tolerance check below outright
  }
  int worst = 0;
  for (std::size_t i = 0; i < lhs.byte_count(); ++i) {
    const int difference = static_cast<int>(lhs.data()[i]) - static_cast<int>(rhs.data()[i]);
    worst = std::max(worst, difference < 0 ? -difference : difference);
  }
  return worst;
}

Image oracle(const Image& input, const std::string& chain_text) {
  const auto chain = imgjit::parse_op_chain(chain_text);
  REQUIRE(chain.has_value());
  return imgjit::cpu::apply_chain(input, *chain);
}

ServerConfig gpu_config() {
  ServerConfig config;
  config.port = 0;  // ephemeral: no fixed port to collide with anything else on the box
  config.num_slots = 4;
  config.slots_per_connection = 2;
  config.max_payload_bytes = 256 * 1024;
  config.queue_capacity = 4;
  config.output_dir = std::string(IMGJIT_TEST_TMP_DIR) + "/phase5_out";
  return config;
}

Server::BackendFactory cuda_factory() {
  return [] { return std::unique_ptr<imgjit::IBackend>(std::make_unique<imgjit::CudaBackend>()); };
}

// What one client thread found out, checked on the main thread afterwards.
struct ClientResult {
  int worst_difference{0};
  std::uint32_t frames_ok{0};
  std::string error;
};

// One connection: pipeline every frame, then match the responses by seq_num and diff
// each against the oracle. Pipelining is deliberate — with 4 slots and a per-connection
// cap of 2, several of these threads park on slot claim, which is the interleaving that
// makes the shared cache and the single worker interesting.
//
// `expected` is computed by the caller on the main thread: the oracle runs through
// parse_op_chain, and checking that with REQUIRE is only legal off a Catch2 worker.
void run_client(std::uint16_t port, const Image& input, const Image& expected,
                const std::string& chain, int frames, ClientResult& result) {
  try {
    Client client("127.0.0.1", port);
    for (int i = 0; i < frames; ++i) {
      client.send(input, chain);
    }
    for (int i = 0; i < frames; ++i) {
      const ClientResponse response = client.receive();
      if (!response.matched) {
        result.error = "a response arrived for a seq_num that was never sent";
        return;
      }
      if (response.status != Status::kOk) {
        result.error = "chain \"" + chain + "\" came back with status " +
                       std::to_string(static_cast<int>(response.status));
        return;
      }
      result.worst_difference =
          std::max(result.worst_difference, max_abs_difference(expected, response.image));
      ++result.frames_ok;
    }
  } catch (const std::exception& error) {
    result.error = error.what();
  }
}

}  // namespace

TEST_CASE("concurrent clients with differing chains all match the CPU oracle", "[gpu-server]") {
  // Odd dimensions, so every kernel runs with a partial thread block on both axes and
  // the bounds check is exercised rather than assumed. Channel counts differ per
  // connection on purpose: `channels` is baked into the kernel, so this is what would
  // catch one connection being served another's compiled chain.
  struct Workload {
    std::string chain;
    int channels;
  };
  const std::vector<Workload> workloads{
      {"grayscale,sobel", 3},
      {"gaussian:1.4", 4},
      {"invert", 1},
      {"grayscale,gaussian:1.4,sobel,threshold:0.3", 3},
      {"brightness:0.2,invert", 4},
      {"", 3},
  };
  constexpr int kWidth = 61;
  constexpr int kHeight = 43;
  constexpr int kFramesPerClient = 6;

  Server server(gpu_config(), cuda_factory());
  server.start();

  std::vector<Image> inputs;
  std::vector<Image> expected;
  std::vector<ClientResult> results(workloads.size());
  inputs.reserve(workloads.size());
  expected.reserve(workloads.size());
  for (std::size_t i = 0; i < workloads.size(); ++i) {
    inputs.push_back(make_image(kWidth, kHeight, workloads[i].channels,
                                0x9e3779b9U + static_cast<std::uint32_t>(i)));
    expected.push_back(oracle(inputs.back(), workloads[i].chain));
  }

  std::vector<std::thread> threads;
  threads.reserve(workloads.size());
  for (std::size_t i = 0; i < workloads.size(); ++i) {
    threads.emplace_back([&, i] {
      run_client(server.port(), inputs[i], expected[i], workloads[i].chain, kFramesPerClient,
                 results[i]);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  for (std::size_t i = 0; i < workloads.size(); ++i) {
    INFO("chain \"" << workloads[i].chain << "\" at " << workloads[i].channels << " channels");
    CHECK(results[i].error.empty());
    CHECK(results[i].frames_ok == kFramesPerClient);
    // <=1 LSB, never bit-equality — FMA contraction and reassociation happen on both
    // sides, so exactness is the wrong bar for anything with a float in it
    // (docs/PLAN.md Phase 2).
    CHECK(results[i].worst_difference <= 1);
  }
  CHECK(server.frames_completed() ==
        static_cast<std::uint64_t>(workloads.size()) * kFramesPerClient);
  CHECK(server.frames_failed() == 0);
}

TEST_CASE("the same connection can switch chains mid-stream", "[gpu-server]") {
  // One connection, many kernel identities: the worker compiles on the first frame of
  // each and hits the cache thereafter, all while the connection stays in sync. This is
  // the single-connection converse of the test above — there, a shared-cache bug shows
  // as crosstalk between clients; here it shows as a frame served by the chain before
  // it.
  Server server(gpu_config(), cuda_factory());
  server.start();

  const Image input = make_image(64, 48, 3, 0xc0ffeeU);
  const std::vector<std::string> chains{"invert",      "grayscale,sobel", "gaussian:0.5",
                                        "threshold:0.4", "invert",        "sobel,invert",
                                        "grayscale,sobel"};

  Client client("127.0.0.1", server.port());
  for (const std::string& chain : chains) {
    client.send(input, chain);
  }
  for (std::size_t i = 0; i < chains.size(); ++i) {
    const ClientResponse response = client.receive();
    REQUIRE(response.matched);
    INFO("chain \"" << response.request.chain << "\"");
    REQUIRE(response.status == Status::kOk);
    CHECK(max_abs_difference(oracle(input, response.request.chain), response.image) <= 1);
  }
  CHECK(client.pending_count() == 0);
}
