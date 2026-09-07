// imgjit-cli — file in, file out, through the real IBackend.
//
//   imgjit-cli --ops "grayscale,sobel" in.png out.png
//
// It could just call cpu::apply_chain directly. It goes through allocate_slots /
// submit / poll_completions instead so that the interface the server will use in
// Phase 4 has an exercised caller from the day it exists, rather than its first user
// being the concurrent one.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "imgjit/backend/cpu/cpu_backend.h"
#include "imgjit/core/kernel_key.h"
#include "imgjit/core/op_chain.h"
#include "imgjit/util/image_io.h"

namespace {

void print_usage() {
  std::fprintf(stderr,
               "usage: imgjit-cli [--ops \"<chain>\"] <input.png> <output.png>\n"
               "  ops: grayscale, invert, brightness:<-1..1>, threshold:<0..1>,\n"
               "       gaussian:<0.1..4>, sobel  (comma-separated, order matters)\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string chain_text;
  std::string input_path;
  std::string output_path;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--ops" && i + 1 < argc) {
      chain_text = argv[++i];
    } else if (argument.rfind("--", 0) == 0) {
      std::fprintf(stderr, "imgjit-cli: unexpected argument: %s\n", argument.c_str());
      print_usage();
      return 2;
    } else if (input_path.empty()) {
      input_path = argument;
    } else if (output_path.empty()) {
      output_path = argument;
    } else {
      std::fprintf(stderr, "imgjit-cli: too many positional arguments\n");
      print_usage();
      return 2;
    }
  }

  if (input_path.empty() || output_path.empty()) {
    print_usage();
    return 2;
  }

  std::string parse_error;
  const auto chain = imgjit::parse_op_chain(chain_text, &parse_error);
  if (!chain.has_value()) {
    std::fprintf(stderr, "imgjit-cli: bad op chain: %s\n", parse_error.c_str());
    return 2;
  }

  try {
    const imgjit::Image input = imgjit::load_png(input_path);

    // The key the Phase 3 cache will be keyed by, printed here because the CLI is the
    // only place a human can see the canonicalizer's output. Note what it is NOT a
    // function of: the image was already loaded, and its dimensions are not an input
    // (CLAUDE.md invariant 4).
    imgjit::KernelKey key;
    key.chain = *chain;
    key.channels = input.channels();
    std::printf("imgjit-cli: %s — %dx%d, %d channels\n", input_path.c_str(), input.width(),
                input.height(), input.channels());
    std::printf("imgjit-cli: chain \"%s\" — kernel key %016llx\n",
                imgjit::canonical_string(*chain).c_str(),
                static_cast<unsigned long long>(imgjit::hash_kernel_key(key)));

    imgjit::CpuBackend backend;
    std::byte* slot = backend.allocate_slots(1, input.byte_count());
    std::memcpy(slot, input.data(), input.byte_count());

    imgjit::FrameJob job;
    job.input = slot;
    job.width = input.width();
    job.height = input.height();
    job.channels = input.channels();
    job.stride = input.stride();
    job.chain = *chain;

    const imgjit::JobHandle handle = backend.submit(job);
    const std::vector<imgjit::Completion> completions = backend.poll_completions();
    if (completions.size() != 1 || completions.front().handle != handle) {
      std::fprintf(stderr, "imgjit-cli: backend returned %zu completions, expected 1\n",
                   completions.size());
      return 1;
    }
    const imgjit::Completion& completion = completions.front();
    if (completion.status != imgjit::JobStatus::kOk) {
      std::fprintf(stderr, "imgjit-cli: processing failed: %s\n", completion.error.c_str());
      return 1;
    }

    imgjit::save_png(output_path, completion.output);
    std::printf("imgjit-cli: wrote %s\n", output_path.c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "imgjit-cli: %s\n", error.what());
    return 1;
  }
}
