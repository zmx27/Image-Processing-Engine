// Phase 1 spike (docs/PLAN.md): inverts a PNG on-GPU twice and diffs both results
// against a scalar CPU loop.
//
//   1a  ahead-of-time PTX from nvcc -> cuModuleLoadData -> cuLaunchKernel
//   1b  the same kernel compiled at runtime by NVRTC instead
//
// Two checkpoints in one binary so a failure is unambiguous about which layer
// broke: if 1a fails the driver plumbing is wrong, if only 1b fails the JIT
// plumbing is. Inversion is an integer pointwise op, so the bar here is *exact*
// equality — no float reassociation is in play.
//
// This is scratch code and is expendable once Phases 2-3 land. What survives is
// src/backend/cuda/{cuda_check,cuda_raii,nvrtc_compile}.h; the oracle below is a
// throwaway loop, deliberately not the Phase 2 CPU backend (which does not exist
// yet).

#include <cuda.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "backend/cuda/cuda_raii.h"
#include "backend/cuda/nvrtc_compile.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace {

constexpr const char* kKernelName = "invert_u8";
constexpr unsigned int kBlockSize = 256;

// The NVRTC twin of tools/invert_aot.cu. Self-contained by necessity: NVRTC has no
// default include path, so this source cannot #include anything at all.
constexpr const char* kInvertSource = R"CUDA(
extern "C" __global__ void invert_u8(unsigned char* data, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    data[i] = (unsigned char)(255 - data[i]);
  }
}
)CUDA";

std::string read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

void write_file(const std::string& path, const std::string& contents) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot write " + path);
  }
  file << contents;
}

std::vector<unsigned char> invert_cpu(const std::vector<unsigned char>& input) {
  std::vector<unsigned char> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = static_cast<unsigned char>(255 - input[i]);
  }
  return output;
}

std::vector<unsigned char> invert_gpu(const imgjit::cuda::CudaModule& module,
                                      const std::vector<unsigned char>& input) {
  const CUfunction kernel = module.get_function(kKernelName);

  imgjit::cuda::DeviceBuffer buffer(input.size());
  CU_CHECK(cuMemcpyHtoD(buffer.get(), input.data(), input.size()));

  CUdeviceptr data = buffer.get();
  int count = static_cast<int>(input.size());
  void* arguments[] = {&data, &count};
  const unsigned int grid = (static_cast<unsigned int>(count) + kBlockSize - 1) / kBlockSize;
  CU_CHECK(cuLaunchKernel(kernel, grid, 1, 1, kBlockSize, 1, 1, 0, nullptr, arguments, nullptr));
  CU_CHECK(cuCtxSynchronize());

  std::vector<unsigned char> output(input.size());
  CU_CHECK(cuMemcpyDtoH(output.data(), buffer.get(), output.size()));
  return output;
}

void report(const char* checkpoint, bool matched) {
  std::printf("imgjit-spike: [%s] vs CPU oracle: %s\n", checkpoint,
              matched ? "exact match" : "MISMATCH");
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string dump_ptx_path;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--out" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (argument == "--dump-ptx" && i + 1 < argc) {
      dump_ptx_path = argv[++i];
    } else if (input_path.empty() && argument.rfind("--", 0) != 0) {
      input_path = argument;
    } else {
      std::fprintf(stderr, "unexpected argument: %s\n", argument.c_str());
      input_path.clear();
      break;
    }
  }

  if (input_path.empty()) {
    std::fprintf(stderr, "usage: imgjit-spike <input.png> [--out <output.png>]"
                         " [--dump-ptx <file.ptx>]\n");
    return 2;
  }

  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* pixels = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
  if (pixels == nullptr) {
    std::fprintf(stderr, "imgjit-spike: cannot load %s: %s\n", input_path.c_str(),
                 stbi_failure_reason());
    return 1;
  }
  const std::vector<unsigned char> input(
      pixels, pixels + static_cast<std::size_t>(width) * height * channels);
  stbi_image_free(pixels);

  std::printf("imgjit-spike: %s — %dx%d, %d channels, %zu bytes\n", input_path.c_str(), width,
              height, channels, input.size());

  const std::vector<unsigned char> expected = invert_cpu(input);

  try {
    const imgjit::cuda::CudaContext context;
    std::printf("imgjit-spike: device 0, %s\n", context.compute_arch().c_str());

    // 1a — ahead-of-time PTX, JIT out of the picture.
    const std::string aot_ptx = read_file(IMGJIT_AOT_PTX_PATH);
    const imgjit::cuda::CudaModule aot_module(aot_ptx.c_str());
    const bool aot_matched = invert_gpu(aot_module, input) == expected;
    report("1a ahead-of-time PTX", aot_matched);

    // 1b — the same kernel, compiled at runtime on top of plumbing 1a just proved.
    const std::string jit_ptx =
        imgjit::cuda::compile_to_ptx(kInvertSource, "invert.cu", context.compute_arch());
    if (!dump_ptx_path.empty()) {
      write_file(dump_ptx_path, jit_ptx);
      std::printf("imgjit-spike: dumped NVRTC PTX to %s (%zu bytes)\n", dump_ptx_path.c_str(),
                  jit_ptx.size());
    }
    const imgjit::cuda::CudaModule jit_module(jit_ptx.c_str());
    const std::vector<unsigned char> jit_result = invert_gpu(jit_module, input);
    const bool jit_matched = jit_result == expected;
    report("1b NVRTC JIT", jit_matched);

    if (!output_path.empty()) {
      if (stbi_write_png(output_path.c_str(), width, height, channels, jit_result.data(),
                         width * channels) == 0) {
        std::fprintf(stderr, "imgjit-spike: cannot write %s\n", output_path.c_str());
        return 1;
      }
      std::printf("imgjit-spike: wrote %s\n", output_path.c_str());
    }

    return (aot_matched && jit_matched) ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "imgjit-spike: %s\n", error.what());
    return 1;
  }
}
