// imgjit-server — the frame server (docs/PLAN.md Phases 4 and 5).
//
//   imgjit-server --port 9000 --slots 8 --slots-per-conn 2 --output-dir out
//   imgjit-server --port 9000 --backend cuda --prewarm "grayscale,gaussian:1.4,sobel"
//
// Phase 4 ran it over the CPU backend, which is the point of Track N: protocol,
// threading and backpressure proven with zero CUDA in the process. Phase 5 changes
// exactly one thing — which backend the factory returns — because the backend is
// constructed BY the worker thread and the server knows nothing else about it.
//
// Both halves of that sentence are load-bearing, and they are why the prewarm below
// lives inside the factory rather than anywhere else: the factory body runs on the
// worker thread, before Server::start() returns, which is the one place a compile can
// happen that satisfies invariant 1 and still blocks startup until it is
// done. A chain warmed there is a cache hit for the first client that asks for it,
// instead of a ~50-200 ms NVRTC stall on a live frame (docs/ARCHITECTURE.md).

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "imgjit/backend/cpu/cpu_backend.h"
#include "imgjit/core/kernel_key.h"
#include "imgjit/core/op_chain.h"
#include "imgjit/net/server.h"

#ifdef IMGJIT_ENABLE_CUDA
#include "backend/cuda/cuda_backend.h"
#endif

namespace {

std::atomic<bool> g_interrupted{false};

extern "C" void handle_interrupt(int /*signal*/) {
  // Async-signal-safe: set a flag and get out. Everything else happens on main().
  g_interrupted.store(true);
}

void print_usage() {
  std::fprintf(stderr,
               "usage: imgjit-server [options]\n"
               "  --port <n>            listen port (0 = ephemeral, printed at startup)\n"
               "  --backend cpu|cuda    default cpu\n"
               "  --streams <n>         CUDA streams kept in flight (default 4; 1 =\n"
               "                        the Phase 5 synchronous baseline)\n"
               "  --tile naive|<n>      stencil kernels: naive (default), or tiled through\n"
               "                        an n x n __shared__ tile, n in 1..32 (cuda only)\n"
               "                        big win on gaussian; no win on sobel, whose radius-1\n"
               "                        apron is too small to be worth tiling (bench/phase7_results.md)\n"
               "  --constants baked|parameterized\n"
               "                        op parameters as kernel literals (default) or launch\n"
               "                        arguments; parameterized needs --tile naive (cuda only)\n"
               "  --prewarm \"<chain>[@ch]\"  compile a chain at startup; repeatable\n"
               "                        (channels default 3; --backend cuda only)\n"
               "  --slots <n>           frame slots in the pool (default 8)\n"
               "  --slots-per-conn <n>  per-connection cap on held slots (default 2)\n"
               "  --max-payload <n>     bytes; doubles as the slot size (default 8 MiB)\n"
               "  --queue <n>           bounded job queue capacity (default 16)\n"
               "  --output-dir <dir>    where flags-bit-1 results are written\n");
}

// One --prewarm argument, already parsed: a bad chain has to be a usage error on main()
// rather than an exception out of the worker thread at startup.
struct PrewarmEntry {
  imgjit::OpChain chain;
  int channels{3};
};

// "grayscale,sobel" or "grayscale,sobel@4". The channel count is part of the kernel
// identity (invariant 4 — it is baked as a literal), so warming a chain warms
// it for ONE channel count and the suffix is how the other two are asked for.
bool parse_prewarm(const std::string& argument, PrewarmEntry& entry) {
  std::string chain_text = argument;
  const std::size_t at = argument.rfind('@');
  if (at != std::string::npos) {
    chain_text = argument.substr(0, at);
    entry.channels = std::atoi(argument.c_str() + at + 1);
    if (!imgjit::is_supported_channel_count(entry.channels)) {
      std::fprintf(stderr, "imgjit-server: --prewarm channels must be 1, 3 or 4\n");
      return false;
    }
  }
  std::string parse_error;
  const auto parsed = imgjit::parse_op_chain(chain_text, &parse_error);
  if (!parsed.has_value()) {
    std::fprintf(stderr, "imgjit-server: bad --prewarm chain: %s\n", parse_error.c_str());
    return false;
  }
  entry.chain = *parsed;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  imgjit::net::ServerConfig config;
  std::string backend_name = "cpu";
  std::vector<PrewarmEntry> prewarm;
  // The Phase 6 A/B axis: --streams 1 is the Phase 5 pipeline (one frame on the GPU at
  // a time), so "measurably faster" is two runs of this binary rather than a comparison
  // against a build that no longer exists.
  std::size_t stream_count = 4;
  // The Phase 7 A/B axis, the same shape as --streams: naive and tiled are two runs of
  // this binary, so the comparison never depends on a build that no longer exists.
  imgjit::TileVariant tile = imgjit::TileVariant::kNaive;
  [[maybe_unused]] int tile_size = 0;  // read only by the CUDA backend
  // The Phase 8 A/B axis, the same shape again.
  imgjit::ConstantsMode constants = imgjit::ConstantsMode::kBaked;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--port" && i + 1 < argc) {
      config.port = static_cast<std::uint16_t>(std::atoi(argv[++i]));
    } else if (argument == "--backend" && i + 1 < argc) {
      backend_name = argv[++i];
    } else if (argument == "--streams" && i + 1 < argc) {
      stream_count = static_cast<std::size_t>(std::atoi(argv[++i]));
    } else if (argument == "--tile" && i + 1 < argc) {
      const std::string value = argv[++i];
      tile = value == "naive" ? imgjit::TileVariant::kNaive : imgjit::TileVariant::kTiled;
      tile_size = value == "naive" ? 0 : std::atoi(value.c_str());
    } else if (argument == "--constants" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (value != "baked" && value != "parameterized") {
        std::fprintf(stderr, "imgjit-server: --constants must be baked or parameterized\n");
        return 2;
      }
      constants = value == "baked" ? imgjit::ConstantsMode::kBaked
                                   : imgjit::ConstantsMode::kParameterized;
    } else if (argument == "--prewarm" && i + 1 < argc) {
      PrewarmEntry entry;
      if (!parse_prewarm(argv[++i], entry)) {
        return 2;
      }
      prewarm.push_back(std::move(entry));
    } else if (argument == "--slots" && i + 1 < argc) {
      config.num_slots = static_cast<std::size_t>(std::atoi(argv[++i]));
    } else if (argument == "--slots-per-conn" && i + 1 < argc) {
      config.slots_per_connection = static_cast<std::size_t>(std::atoi(argv[++i]));
    } else if (argument == "--max-payload" && i + 1 < argc) {
      config.max_payload_bytes = static_cast<std::uint32_t>(std::atol(argv[++i]));
    } else if (argument == "--queue" && i + 1 < argc) {
      config.queue_capacity = static_cast<std::size_t>(std::atoi(argv[++i]));
    } else if (argument == "--output-dir" && i + 1 < argc) {
      config.output_dir = argv[++i];
    } else {
      std::fprintf(stderr, "imgjit-server: unexpected argument: %s\n", argument.c_str());
      print_usage();
      return 2;
    }
  }

  if (backend_name != "cpu" && backend_name != "cuda") {
    std::fprintf(stderr, "imgjit-server: unknown backend \"%s\" (expected cpu or cuda)\n",
                 backend_name.c_str());
    return 2;
  }
#ifndef IMGJIT_ENABLE_CUDA
  if (backend_name == "cuda") {
    std::fprintf(stderr,
                 "imgjit-server: this build has no CUDA backend — it was configured with no\n"
                 "               CUDA toolkit present.\n");
    return 2;
  }
#endif
  if (!prewarm.empty() && backend_name != "cuda") {
    std::fprintf(stderr, "imgjit-server: --prewarm needs --backend cuda (nothing else compiles)\n");
    return 2;
  }
  if (stream_count == 0) {
    std::fprintf(stderr, "imgjit-server: --streams must be at least 1\n");
    return 2;
  }
  if (tile != imgjit::TileVariant::kNaive && backend_name != "cuda") {
    std::fprintf(stderr, "imgjit-server: --tile needs --backend cuda (nothing else tiles)\n");
    return 2;
  }
  if (constants != imgjit::ConstantsMode::kBaked && backend_name != "cuda") {
    std::fprintf(stderr,
                 "imgjit-server: --constants needs --backend cuda (nothing else compiles)\n");
    return 2;
  }
#ifdef IMGJIT_ENABLE_CUDA
  if (!imgjit::cuda::is_supported_tile(tile, tile_size)) {
    std::fprintf(stderr, "imgjit-server: --tile must be naive or a tile edge in 1..%d\n",
                 imgjit::cuda::kMaxTileSize);
    return 2;
  }
  if (!imgjit::cuda::is_supported_constants(constants, tile)) {
    std::fprintf(stderr, "imgjit-server: --constants parameterized needs --tile naive\n");
    return 2;
  }
#endif

  // Runs ON the worker thread, and start() does not return until it has finished — so a
  // missing GPU and a failed prewarm compile both surface as an exception from start()
  // rather than as a server that accepts connections it cannot serve.
  imgjit::net::Server::BackendFactory factory = [] {
    return std::unique_ptr<imgjit::IBackend>(std::make_unique<imgjit::CpuBackend>());
  };
#ifdef IMGJIT_ENABLE_CUDA
  if (backend_name == "cuda") {
    factory = [&prewarm, stream_count, tile, tile_size, constants] {
      auto backend =
          std::make_unique<imgjit::CudaBackend>(0, stream_count, tile, tile_size, constants);
      const std::string tile_text =
          tile == imgjit::TileVariant::kNaive
              ? std::string("naive")
              : std::to_string(tile_size) + "x" + std::to_string(tile_size) + " tiled";
      std::printf("imgjit-server: device 0, %s, %zu stream%s, %s stencils, %s constants\n",
                  backend->context().compute_arch().c_str(), backend->stream_count(),
                  backend->stream_count() == 1 ? "" : "s", tile_text.c_str(),
                  constants == imgjit::ConstantsMode::kBaked ? "baked" : "parameterized");
      if (!prewarm.empty()) {
        const auto started = std::chrono::steady_clock::now();
        for (const PrewarmEntry& entry : prewarm) {
          backend->prewarm(entry.chain, entry.channels);
        }
        const std::chrono::duration<double, std::milli> elapsed =
            std::chrono::steady_clock::now() - started;
        std::printf("imgjit-server: prewarmed %zu chain%s in %.1f ms — %zu NVRTC compile%s\n",
                    prewarm.size(), prewarm.size() == 1 ? "" : "s", elapsed.count(),
                    backend->compile_count(), backend->compile_count() == 1 ? "" : "s");
      }
      return std::unique_ptr<imgjit::IBackend>(std::move(backend));
    };
  }
#endif

  std::signal(SIGINT, handle_interrupt);
  std::signal(SIGTERM, handle_interrupt);

  try {
    imgjit::net::Server server(config, factory);
    server.start();
    std::printf("imgjit-server: listening on 127.0.0.1:%u — %s backend, %zu slots x %u bytes, "
                "%zu per conn\n",
                static_cast<unsigned>(server.port()), backend_name.c_str(), config.num_slots,
                config.max_payload_bytes, config.slots_per_connection);
    std::fflush(stdout);

    while (!g_interrupted.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    server.stop();
    std::printf("imgjit-server: %llu connections, %llu frames completed, %llu failed, "
                "%llu slot waits, queue high-water %zu\n",
                static_cast<unsigned long long>(server.connections_accepted()),
                static_cast<unsigned long long>(server.frames_completed()),
                static_cast<unsigned long long>(server.frames_failed()),
                static_cast<unsigned long long>(server.slot_waits()), server.max_queue_depth());

    // Read after stop(), which is when it becomes meaningful: the worker snapshots it
    // on its way out (docs/PLAN.md Phase 6, "instrument queue depth, occupancy, stall
    // counts"). mean_in_flight near 1.0 with --streams 4 means the pipeline serialized
    // and the streams bought nothing, whatever the throughput number says.
    //
    // The mean kernel time is Phase 7's instrument, and a clean per-frame number only
    // under --streams 1 (imgjit/backend/backend.h).
    const imgjit::BackendStats stats = server.backend_stats();
    std::printf("imgjit-server: streams — mean in flight %.2f, peak %llu, %llu submit stalls, "
                "%llu context recreation%s, mean kernel %.3f ms, %llu NVRTC compiles\n",
                stats.mean_in_flight, static_cast<unsigned long long>(stats.max_in_flight),
                static_cast<unsigned long long>(stats.submit_stalls),
                static_cast<unsigned long long>(stats.context_recreations),
                stats.context_recreations == 1 ? "" : "s", stats.mean_kernel_ms,
                static_cast<unsigned long long>(stats.jit_compiles));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "imgjit-server: %s\n", error.what());
    return 1;
  }
}
