// imgjit-server — the frame server (docs/PLAN.md Phase 4).
//
//   imgjit-server --port 9000 --slots 8 --slots-per-conn 2 --output-dir out
//
// Phase 4 runs it over the CPU backend, which is the point of Track N: protocol,
// threading and backpressure are proven with zero CUDA in the process. Phase 5 changes
// exactly one thing here — the factory below returns a CudaBackend — because the
// backend is constructed by the worker thread and the server knows nothing else about
// it.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <thread>

#include "imgjit/backend/cpu/cpu_backend.h"
#include "imgjit/net/server.h"

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
               "  --slots <n>           frame slots in the pool (default 8)\n"
               "  --slots-per-conn <n>  per-connection cap on held slots (default 2)\n"
               "  --max-payload <n>     bytes; doubles as the slot size (default 8 MiB)\n"
               "  --queue <n>           bounded job queue capacity (default 16)\n"
               "  --output-dir <dir>    where flags-bit-1 results are written\n");
}

}  // namespace

int main(int argc, char** argv) {
  imgjit::net::ServerConfig config;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--port" && i + 1 < argc) {
      config.port = static_cast<std::uint16_t>(std::atoi(argv[++i]));
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

  std::signal(SIGINT, handle_interrupt);
  std::signal(SIGTERM, handle_interrupt);

  try {
    imgjit::net::Server server(config, [] { return std::make_unique<imgjit::CpuBackend>(); });
    server.start();
    std::printf("imgjit-server: listening on 127.0.0.1:%u — %zu slots x %u bytes, %zu per conn\n",
                static_cast<unsigned>(server.port()), config.num_slots, config.max_payload_bytes,
                config.slots_per_connection);
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
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "imgjit-server: %s\n", error.what());
    return 1;
  }
}
