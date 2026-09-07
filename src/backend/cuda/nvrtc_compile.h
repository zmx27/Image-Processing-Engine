#pragma once

// Runtime compilation of a self-contained CUDA source string to PTX.
//
// NVRTC has no default include path, so `source` must not #include anything — the
// stable device helpers arrive in Phase 3 as a raw-string prelude prepended here,
// not as headers on disk (docs/ARCHITECTURE.md).

#include <nvrtc.h>

#include <cstddef>
#include <string>
#include <utility>

#include "backend/cuda/cuda_check.h"

namespace imgjit::cuda {

namespace detail {

class NvrtcProgram {
 public:
  NvrtcProgram(const std::string& source, const char* name) {
    NVRTC_CHECK(nvrtcCreateProgram(&program_, source.c_str(), name, 0, nullptr, nullptr));
  }

  ~NvrtcProgram() {
    if (program_ != nullptr) {
      nvrtcDestroyProgram(&program_);
    }
  }

  NvrtcProgram(const NvrtcProgram&) = delete;
  NvrtcProgram& operator=(const NvrtcProgram&) = delete;

  nvrtcProgram get() const { return program_; }

 private:
  nvrtcProgram program_{};
};

// NVRTC's *Size getters count the trailing NUL; the string must own that byte while
// the getter writes, and drop it afterwards.
inline std::string read_sized(std::size_t size_with_nul, auto&& getter) {
  if (size_with_nul <= 1) {
    return {};
  }
  std::string text(size_with_nul, '\0');
  getter(text.data());
  text.pop_back();
  return text;
}

}  // namespace detail

// Returns PTX for `source`. Throws CudaError with the NVRTC build log attached if
// compilation fails — the log is the only useful diagnostic for generated code.
inline std::string compile_to_ptx(const std::string& source, const char* program_name,
                                  const std::string& gpu_architecture) {
  const detail::NvrtcProgram program(source, program_name);

  const std::string arch_option = "--gpu-architecture=" + gpu_architecture;
  const char* options[] = {arch_option.c_str()};
  const nvrtcResult compile_result = nvrtcCompileProgram(program.get(), 1, options);

  std::size_t log_size = 0;
  NVRTC_CHECK(nvrtcGetProgramLogSize(program.get(), &log_size));
  const std::string log = detail::read_sized(
      log_size, [&](char* out) { NVRTC_CHECK(nvrtcGetProgramLog(program.get(), out)); });

  if (compile_result != NVRTC_SUCCESS) {
    throw CudaError(std::string("NVRTC failed to compile ") + program_name + " for " +
                    gpu_architecture + ":\n" + log);
  }

  std::size_t ptx_size = 0;
  NVRTC_CHECK(nvrtcGetPTXSize(program.get(), &ptx_size));
  return detail::read_sized(ptx_size,
                            [&](char* out) { NVRTC_CHECK(nvrtcGetPTX(program.get(), out)); });
}

}  // namespace imgjit::cuda
