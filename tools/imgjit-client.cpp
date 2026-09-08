// imgjit-client — the reference client (docs/PLAN.md Phase 4).
//
//   imgjit-client --port 9000 --ops "grayscale,sobel" in.png out.png
//   imgjit-client --port 9000 --ops "gaussian:1.4" --repeat 8 --server-write in.png
//
// It PIPELINES by default: every frame is sent before any response is read. That is the
// load that deadlocks a server which releases a frame slot only after writing the
// response (docs/ARCHITECTURE.md decision 6), so making it the default here means the
// interesting case is the one that gets run by hand.
//
// Responses are matched by seq_num through the client's pending map, never by arrival
// order — see include/imgjit/net/client.h.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "imgjit/net/client.h"
#include "imgjit/util/image_io.h"

namespace {

void print_usage() {
  std::fprintf(stderr,
               "usage: imgjit-client [options] <input.png> [output.png]\n"
               "  --host <ip>      default 127.0.0.1\n"
               "  --port <n>       required\n"
               "  --ops \"<chain>\"  op chain to request\n"
               "  --repeat <n>     send n frames before reading any response (default 1)\n"
               "  --no-echo        do not ask for the result back over the wire\n"
               "  --server-write   also ask the server to write the result as a PNG\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string host = "127.0.0.1";
  std::uint16_t port = 0;
  std::string chain_text;
  std::string input_path;
  std::string output_path;
  int repeat = 1;
  bool echo = true;
  bool server_write = false;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--host" && i + 1 < argc) {
      host = argv[++i];
    } else if (argument == "--port" && i + 1 < argc) {
      port = static_cast<std::uint16_t>(std::atoi(argv[++i]));
    } else if (argument == "--ops" && i + 1 < argc) {
      chain_text = argv[++i];
    } else if (argument == "--repeat" && i + 1 < argc) {
      repeat = std::atoi(argv[++i]);
    } else if (argument == "--no-echo") {
      echo = false;
    } else if (argument == "--server-write") {
      server_write = true;
    } else if (argument.rfind("--", 0) == 0) {
      std::fprintf(stderr, "imgjit-client: unexpected argument: %s\n", argument.c_str());
      print_usage();
      return 2;
    } else if (input_path.empty()) {
      input_path = argument;
    } else if (output_path.empty()) {
      output_path = argument;
    } else {
      std::fprintf(stderr, "imgjit-client: too many positional arguments\n");
      return 2;
    }
  }

  if (port == 0 || input_path.empty() || repeat < 1) {
    print_usage();
    return 2;
  }

  const std::uint8_t flags = static_cast<std::uint8_t>((echo ? imgjit::net::kFlagEcho : 0) |
                                                       (server_write ? imgjit::net::kFlagServerWrite
                                                                     : 0));

  try {
    const imgjit::Image input = imgjit::load_png(input_path);
    imgjit::net::Client client(host, port);

    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
      client.send(input, chain_text, flags);
    }

    imgjit::Image first_result;
    int failures = 0;
    for (int i = 0; i < repeat; ++i) {
      const imgjit::net::ClientResponse response = client.receive();
      if (!response.matched) {
        std::fprintf(stderr, "imgjit-client: response for seq %u was never sent\n",
                     response.seq_num);
        return 1;
      }
      if (response.status != imgjit::net::Status::kOk) {
        std::fprintf(stderr, "imgjit-client: seq %u failed: %s\n", response.seq_num,
                     std::string(imgjit::net::status_message(response.status)).c_str());
        ++failures;
        continue;
      }
      if (first_result.empty()) {
        first_result = response.image;
      }
    }
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - started;

    std::printf("imgjit-client: %d frame%s of %dx%d/%dch, chain \"%s\" — %.2f ms total, "
                "%.2f ms/frame, %d failed\n",
                repeat, repeat == 1 ? "" : "s", input.width(), input.height(), input.channels(),
                chain_text.c_str(), elapsed.count(), elapsed.count() / repeat, failures);

    if (!output_path.empty() && !first_result.empty()) {
      imgjit::save_png(output_path, first_result);
      std::printf("imgjit-client: wrote %s\n", output_path.c_str());
    }
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "imgjit-client: %s\n", error.what());
    return 1;
  }
}
