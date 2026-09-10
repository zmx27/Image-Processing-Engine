// imgjit-bench — the timing harness (docs/PLAN.md Phase 5).
//
//   imgjit-bench --port 9000 --connections 4 --frames 200 --width 1024 --height 1024
//                --ops "grayscale,gaussian:1.4,sobel" --csv bench/baseline_phase5.csv
//
// Phases 6 and 7 are both gated on being "measurably faster", which is only a claim if
// something recorded a number first. This is that something, and Phase 8's CSV matrix
// widens it rather than reinventing the measurement — which is why the row it appends
// is already labelled by every axis Phase 8 sweeps.
//
// CLOSED LOOP, NOT A FIRE HOSE. Each connection keeps at most `--window` frames in
// flight and sends a replacement only when a response comes back. The alternative —
// send everything, then read everything, which is what imgjit-client --repeat does —
// measures something real (it is the load that deadlocks a server releasing slots after
// the write) but it is useless for latency: every frame after the first few reports the
// queueing delay of the whole run rather than a round trip. Keep both; they answer
// different questions.
//
// EACH CONNECTION'S FIRST FRAME IS REPORTED SEPARATELY and excluded from the
// percentiles. On the CUDA backend that frame pays the ~50-200 ms NVRTC compile unless
// the chain was prewarmed, so folding it in would put a compile in the p99 of a run
// measuring execution. Reported rather than dropped, because the gap between it and p50
// IS the cache's headline number, and `imgjit-server --prewarm` closing that gap is how
// docs/ARCHITECTURE.md's prewarm claim maps to something measured.
//
// Portable: no CUDA. It drives the server over a socket, so which backend is on the
// other end is exactly the variable being measured.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "imgjit/core/image.h"
#include "imgjit/net/client.h"

namespace {

void print_usage() {
  std::fprintf(stderr,
               "usage: imgjit-bench [options]\n"
               "  --host <ip>          default 127.0.0.1\n"
               "  --port <n>           required\n"
               "  --connections <n>    concurrent clients (default 1)\n"
               "  --frames <n>         frames per connection (default 100)\n"
               "  --window <n>         frames in flight per connection (default 2)\n"
               "  --width/--height <n> synthetic frame size (default 512x512)\n"
               "  --channels <1|3|4>   default 3\n"
               "  --ops \"<chain>\"      repeatable; connection i uses chain i %% count\n"
               "  --no-echo            do not ask for the result back over the wire\n"
               "  --label <name>       tag for the CSV row (default \"run\")\n"
               "  --csv <file>         append one row, writing the header if new\n");
}

// Deterministic, so two runs measure the same work and a baseline is comparable to the
// run it is being compared against. Not loaded from a PNG: Phase 8 sweeps 512" to 4K
// and there is no reason to check in fixtures for sizes only the benchmark uses.
imgjit::Image synthetic_frame(int width, int height, int channels) {
  imgjit::Image image(width, height, channels);
  std::uint32_t state = 0x9e3779b9U;
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

struct ConnectionResult {
  std::vector<double> latencies_ms;  // every frame but the first
  double first_frame_ms{0.0};
  std::uint64_t failures{0};
  std::string error;  // non-empty if the connection died outright
};

// Nearest-rank, on an already-sorted sample. No interpolation: with a few hundred
// points the difference is noise, and a percentile someone can recompute by hand from
// the raw numbers is worth more in a baseline than a marginally better estimator.
double percentile(const std::vector<double>& sorted, double fraction) {
  if (sorted.empty()) {
    return 0.0;
  }
  const auto rank = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(
                                                                      sorted.size())));
  const std::size_t index = rank == 0 ? 0 : rank - 1;
  return sorted[std::min(index, sorted.size() - 1)];
}

double milliseconds_since(std::chrono::steady_clock::time_point started) {
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - started;
  return elapsed.count();
}

void run_connection(const std::string& host, std::uint16_t port, const imgjit::Image& frame,
                    const std::string& chain, int frames, int window, std::uint8_t flags,
                    ConnectionResult& result) {
  try {
    imgjit::net::Client client(host, port);
    int sent = 0;
    int received = 0;

    const int initial = std::min(window, frames);
    for (int i = 0; i < initial; ++i) {
      client.send(frame, chain, flags);
      ++sent;
    }

    while (received < frames) {
      const imgjit::net::ClientResponse response = client.receive();
      const double latency_ms = response.matched
                                    ? milliseconds_since(response.request.sent_at)
                                    : 0.0;
      ++received;

      if (!response.matched) {
        // A seq_num this client never sent means the connection desynchronized, which
        // makes every number after it meaningless — stop rather than average it in.
        result.error = "server answered a seq_num that was never sent";
        return;
      }
      if (response.status != imgjit::net::Status::kOk) {
        ++result.failures;
      } else if (received == 1) {
        result.first_frame_ms = latency_ms;
      } else {
        result.latencies_ms.push_back(latency_ms);
      }

      if (sent < frames) {
        client.send(frame, chain, flags);
        ++sent;
      }
    }
  } catch (const std::exception& error) {
    result.error = error.what();
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string host = "127.0.0.1";
  std::uint16_t port = 0;
  int connections = 1;
  int frames = 100;
  int window = 2;
  int width = 512;
  int height = 512;
  int channels = 3;
  bool echo = true;
  std::string label = "run";
  std::string csv_path;
  std::vector<std::string> chains;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--host" && i + 1 < argc) {
      host = argv[++i];
    } else if (argument == "--port" && i + 1 < argc) {
      port = static_cast<std::uint16_t>(std::atoi(argv[++i]));
    } else if (argument == "--connections" && i + 1 < argc) {
      connections = std::atoi(argv[++i]);
    } else if (argument == "--frames" && i + 1 < argc) {
      frames = std::atoi(argv[++i]);
    } else if (argument == "--window" && i + 1 < argc) {
      window = std::atoi(argv[++i]);
    } else if (argument == "--width" && i + 1 < argc) {
      width = std::atoi(argv[++i]);
    } else if (argument == "--height" && i + 1 < argc) {
      height = std::atoi(argv[++i]);
    } else if (argument == "--channels" && i + 1 < argc) {
      channels = std::atoi(argv[++i]);
    } else if (argument == "--ops" && i + 1 < argc) {
      chains.emplace_back(argv[++i]);
    } else if (argument == "--no-echo") {
      echo = false;
    } else if (argument == "--label" && i + 1 < argc) {
      label = argv[++i];
    } else if (argument == "--csv" && i + 1 < argc) {
      csv_path = argv[++i];
    } else {
      std::fprintf(stderr, "imgjit-bench: unexpected argument: %s\n", argument.c_str());
      print_usage();
      return 2;
    }
  }

  if (port == 0 || connections < 1 || frames < 1 || window < 1 || width < 1 || height < 1 ||
      !imgjit::is_supported_channel_count(channels)) {
    print_usage();
    return 2;
  }
  if (chains.empty()) {
    chains.emplace_back("");
  }

  const std::uint8_t flags = echo ? imgjit::net::kFlagEcho : 0;

  try {
    // One image, shared by every connection: it is read-only once built, and building
    // a 4K frame per thread would put the generator in the measurement.
    const imgjit::Image frame = synthetic_frame(width, height, channels);
    const std::size_t payload_bytes = frame.byte_count();

    std::printf("imgjit-bench: %d connection%s x %d frames of %dx%d/%dch, window %d, echo %s\n",
                connections, connections == 1 ? "" : "s", frames, width, height, channels, window,
                echo ? "on" : "off");
    for (std::size_t i = 0; i < chains.size(); ++i) {
      std::printf("imgjit-bench:   chain[%zu] \"%s\"\n", i, chains[i].c_str());
    }
    std::fflush(stdout);

    std::vector<ConnectionResult> results(static_cast<std::size_t>(connections));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(connections));

    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < connections; ++i) {
      const std::string& chain = chains[static_cast<std::size_t>(i) % chains.size()];
      threads.emplace_back([&, i, chain] {
        run_connection(host, port, frame, chain, frames, window, flags,
                       results[static_cast<std::size_t>(i)]);
      });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
    const double elapsed_ms = milliseconds_since(started);

    std::vector<double> latencies;
    double cold_first_ms = 0.0;
    std::uint64_t failures = 0;
    bool broken = false;
    for (const ConnectionResult& result : results) {
      if (!result.error.empty()) {
        std::fprintf(stderr, "imgjit-bench: connection failed: %s\n", result.error.c_str());
        broken = true;
        continue;
      }
      latencies.insert(latencies.end(), result.latencies_ms.begin(), result.latencies_ms.end());
      cold_first_ms = std::max(cold_first_ms, result.first_frame_ms);
      failures += result.failures;
    }
    if (broken) {
      return 1;
    }

    std::sort(latencies.begin(), latencies.end());
    const double p50 = percentile(latencies, 0.50);
    const double p99 = percentile(latencies, 0.99);
    double mean = 0.0;
    for (const double value : latencies) {
      mean += value;
    }
    mean = latencies.empty() ? 0.0 : mean / static_cast<double>(latencies.size());
    const double max_ms = latencies.empty() ? 0.0 : latencies.back();

    const double total_frames = static_cast<double>(connections) * static_cast<double>(frames);
    const double fps = elapsed_ms > 0.0 ? total_frames / (elapsed_ms / 1000.0) : 0.0;
    // Request payload only. The echo doubles the bytes actually on the wire, so this is
    // deliberately the smaller, unambiguous number rather than one that silently
    // depends on --no-echo.
    const double req_mb_per_s =
        elapsed_ms > 0.0 ? (total_frames * static_cast<double>(payload_bytes)) /
                               (elapsed_ms / 1000.0) / (1024.0 * 1024.0)
                         : 0.0;

    std::printf("imgjit-bench: %.0f frames in %.1f ms — %.1f fps, %.1f MB/s request payload\n",
                total_frames, elapsed_ms, fps, req_mb_per_s);
    std::printf("imgjit-bench: round trip over %zu frames (each connection's first excluded)\n",
                latencies.size());
    std::printf("imgjit-bench:   p50 %.3f ms   p99 %.3f ms   mean %.3f ms   max %.3f ms\n", p50,
                p99, mean, max_ms);
    std::printf("imgjit-bench: first frame per connection: %.3f ms worst — cold NVRTC compile "
                "unless the chain was prewarmed\n",
                cold_first_ms);
    if (failures > 0) {
      std::fprintf(stderr, "imgjit-bench: %llu frames came back with an error status\n",
                   static_cast<unsigned long long>(failures));
    }

    if (!csv_path.empty()) {
      const bool exists = std::ifstream(csv_path).good();
      std::ofstream csv(csv_path, std::ios::app);
      if (!csv) {
        std::fprintf(stderr, "imgjit-bench: cannot write %s\n", csv_path.c_str());
        return 1;
      }
      if (!exists) {
        csv << "label,connections,window,frames_per_conn,width,height,channels,chains,fps,"
               "req_mb_per_s,p50_ms,p99_ms,mean_ms,max_ms,cold_first_ms,failures\n";
      }
      csv << label << ',' << connections << ',' << window << ',' << frames << ',' << width << ','
          << height << ',' << channels << ',' << chains.size() << ',' << fps << ','
          << req_mb_per_s << ',' << p50 << ',' << p99 << ',' << mean << ',' << max_ms << ','
          << cold_first_ms << ',' << failures << '\n';
      std::printf("imgjit-bench: appended row \"%s\" to %s\n", label.c_str(), csv_path.c_str());
    }

    return failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "imgjit-bench: %s\n", error.what());
    return 1;
  }
}
