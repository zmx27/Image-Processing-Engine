// imgjit-cli — file in, file out, through the real IBackend.
//
//   imgjit-cli --ops "grayscale,sobel" in.png out.png
//   imgjit-cli --ops "grayscale,gaussian:1.4,sobel" --backend cuda
//              --dump-source chain.cu --repeat 5 in.png out.png
//
// It could just call cpu::apply_chain directly. It goes through allocate_slots /
// submit / poll_completions instead so that the interface the server will use in
// Phase 4 has an exercised caller from the day it exists, rather than its first user
// being the concurrent one. Phase 3 gets that for free: `--backend cuda` swaps the
// implementation and nothing else here changes, which is the same swap Phase 5 makes
// inside the server.
//
// Builds on macOS with no CUDA toolkit present: `--backend cuda` is compiled in only
// when one was found, but `--dump-source` works either way — codegen is portable, so
// the generated kernel can be read and diffed locally before it is ever compiled.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/cuda/codegen.h"
#include "imgjit/backend/cpu/cpu_backend.h"
#include "imgjit/core/kernel_key.h"
#include "imgjit/core/op_chain.h"
#include "imgjit/util/image_io.h"

#ifdef IMGJIT_ENABLE_CUDA
#include "backend/cuda/cuda_backend.h"
#endif

namespace {

void print_usage() {
  std::fprintf(stderr,
               "usage: imgjit-cli [options] <input.png> <output.png>\n"
               "  --ops \"<chain>\"      grayscale, invert, brightness:<-1..1>,\n"
               "                       threshold:<0..1>, gaussian:<0.1..4>, sobel\n"
               "                       (comma-separated, order matters)\n"
               "  --backend cpu|cuda   default cpu\n"
               "  --dump-source <file> write the generated CUDA source (works without a GPU)\n"
               "  --dump-ptx <file>    write the PTX NVRTC produced (--backend cuda only)\n"
               "  --repeat <n>         run the chain n times; run 1 is cold, the rest warm\n");
}

void write_file(const std::string& path, const std::string& contents) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot write " + path);
  }
  file << contents;
}

double milliseconds_since(std::chrono::steady_clock::time_point started) {
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - started;
  return elapsed.count();
}

}  // namespace

int main(int argc, char** argv) {
  std::string chain_text;
  std::string backend_name = "cpu";
  std::string input_path;
  std::string output_path;
  std::string dump_source_path;
  std::string dump_ptx_path;
  int repeat = 1;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--ops" && i + 1 < argc) {
      chain_text = argv[++i];
    } else if (argument == "--backend" && i + 1 < argc) {
      backend_name = argv[++i];
    } else if (argument == "--dump-source" && i + 1 < argc) {
      dump_source_path = argv[++i];
    } else if (argument == "--dump-ptx" && i + 1 < argc) {
      dump_ptx_path = argv[++i];
    } else if (argument == "--repeat" && i + 1 < argc) {
      repeat = std::atoi(argv[++i]);
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
  if (repeat < 1) {
    std::fprintf(stderr, "imgjit-cli: --repeat must be at least 1\n");
    return 2;
  }
  if (backend_name != "cpu" && backend_name != "cuda") {
    std::fprintf(stderr, "imgjit-cli: unknown backend \"%s\" (expected cpu or cuda)\n",
                 backend_name.c_str());
    return 2;
  }
#ifndef IMGJIT_ENABLE_CUDA
  if (backend_name == "cuda") {
    std::fprintf(stderr,
                 "imgjit-cli: this build has no CUDA backend — it was configured with no CUDA\n"
                 "            toolkit present. --dump-source still works.\n");
    return 2;
  }
#endif

  std::string parse_error;
  const auto chain = imgjit::parse_op_chain(chain_text, &parse_error);
  if (!chain.has_value()) {
    std::fprintf(stderr, "imgjit-cli: bad op chain: %s\n", parse_error.c_str());
    return 2;
  }

  try {
    const imgjit::Image input = imgjit::load_png(input_path);

    // The key the kernel cache is keyed by, printed here because the CLI is the only
    // place a human can see the canonicalizer's output. Note what it is NOT a function
    // of: the image was already loaded, and its dimensions are not an input (CLAUDE.md
    // invariant 4).
    imgjit::KernelKey key;
    key.chain = *chain;
    key.channels = input.channels();
    std::printf("imgjit-cli: %s — %dx%d, %d channels\n", input_path.c_str(), input.width(),
                input.height(), input.channels());
    std::printf("imgjit-cli: chain \"%s\" — kernel key %016llx, backend %s\n",
                imgjit::canonical_string(*chain).c_str(),
                static_cast<unsigned long long>(imgjit::hash_kernel_key(key)),
                backend_name.c_str());

    if (!dump_source_path.empty()) {
      const imgjit::cuda::GeneratedProgram program = imgjit::cuda::emit_cuda_source(key);
      write_file(dump_source_path, program.source);
      std::printf("imgjit-cli: dumped generated source to %s (%zu bytes, %zu stage%s)\n",
                  dump_source_path.c_str(), program.source.size(), program.stages.size(),
                  program.stages.size() == 1 ? "" : "s");
    }

    std::unique_ptr<imgjit::IBackend> backend;
#ifdef IMGJIT_ENABLE_CUDA
    imgjit::CudaBackend* cuda_backend = nullptr;
    if (backend_name == "cuda") {
      auto owned = std::make_unique<imgjit::CudaBackend>();
      cuda_backend = owned.get();
      std::printf("imgjit-cli: device 0, %s\n", owned->context().compute_arch().c_str());
      backend = std::move(owned);
    }
#endif
    if (backend == nullptr) {
      backend = std::make_unique<imgjit::CpuBackend>();
    }

    std::byte* slot = backend->allocate_slots(1, input.byte_count());
    std::memcpy(slot, input.data(), input.byte_count());

    imgjit::FrameJob job;
    job.input = slot;
    job.width = input.width();
    job.height = input.height();
    job.channels = input.channels();
    job.stride = input.stride();
    job.chain = *chain;

    // Run 1 pays the NVRTC compile, runs 2..n hit the cache. That difference is the
    // whole point of the cache, so the CLI reports it rather than leaving it to be
    // inferred (docs/PLAN.md Phase 3, "measure cold-compile vs. warm-hit latency").
    //
    // Submit then POLL UNTIL IT COMES BACK, rather than assuming one poll suffices. The
    // CPU backend completes inside submit() and the Phase 5 CUDA backend did too, but
    // from Phase 6 the CUDA one returns while the frame is still on the GPU — which is
    // the interface's contract as written since Phase 2, and this loop is what an
    // honest caller of it looks like. One frame in flight means the loop spins at most
    // as long as that frame takes.
    imgjit::Image result;
    for (int iteration = 0; iteration < repeat; ++iteration) {
      const auto started = std::chrono::steady_clock::now();
      const imgjit::JobHandle handle = backend->submit(job);
      std::vector<imgjit::Completion> completions;
      while (completions.empty()) {
        completions = backend->poll_completions();
      }
      const double elapsed_ms = milliseconds_since(started);

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
      if (repeat > 1) {
        std::printf("imgjit-cli: run %d/%d — %.3f ms (%s)\n", iteration + 1, repeat, elapsed_ms,
                    iteration == 0 ? "cold" : "warm");
      }
      result = completion.output;
    }

#ifdef IMGJIT_ENABLE_CUDA
    if (cuda_backend != nullptr) {
      std::printf("imgjit-cli: NVRTC compiles: %zu\n", cuda_backend->compile_count());
      if (!dump_ptx_path.empty() && !chain->ops.empty()) {
        write_file(dump_ptx_path, cuda_backend->cache().get(key).ptx);
        std::printf("imgjit-cli: dumped PTX to %s\n", dump_ptx_path.c_str());
      }
    }
#endif

    imgjit::save_png(output_path, result);
    std::printf("imgjit-cli: wrote %s\n", output_path.c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "imgjit-cli: %s\n", error.what());
    return 1;
  }
}
