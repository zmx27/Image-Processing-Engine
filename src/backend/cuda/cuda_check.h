#pragma once

// Error-checking wrappers for the CUDA Driver API and NVRTC. Every driver/NVRTC
// call in this project goes through CU_CHECK / NVRTC_CHECK; there are no unchecked
// calls.
//
// This directory is the only place permitted to include <cuda.h> / <nvrtc.h>
// (invariant 1).

#include <cuda.h>
#include <nvrtc.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace imgjit::cuda {

class CudaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// A COMPILE failure, not a driver failure, and the distinction is load-bearing from
// Phase 6 on. The backend treats any CudaError out of the driver as potentially sticky
// and recovers by recreating the whole context, because that is the only way back from
// an illegal access. NVRTC touches no context at all — a chain that fails to compile is
// an ordinary bad job, and tearing the GPU down over one would turn a client's typo
// into every other client's failed frame.
class NvrtcError : public CudaError {
 public:
  using CudaError::CudaError;
};

namespace detail {

inline std::string format_site(const char* expr, const char* file, int line) {
  return std::string(file) + ":" + std::to_string(line) + ": " + expr;
}

inline void cu_check(CUresult result, const char* expr, const char* file, int line) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw CudaError(format_site(expr, file, line) + " failed: " + (name ? name : "UNKNOWN") + " (" +
                  (description ? description : "no description") + ")");
}

inline void nvrtc_check(nvrtcResult result, const char* expr, const char* file, int line) {
  if (result == NVRTC_SUCCESS) {
    return;
  }
  throw NvrtcError(format_site(expr, file, line) + " failed: " + nvrtcGetErrorString(result));
}

// Destructors must not throw, so RAII teardown reports and swallows instead. Only
// ever used on the destruction path.
inline void cu_check_nothrow(CUresult result, const char* expr, const char* file,
                             int line) noexcept {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  std::fprintf(stderr, "imgjit: ignoring failure during teardown: %s:%d: %s -> %s\n", file, line,
               expr, name ? name : "UNKNOWN");
}

}  // namespace detail

#define CU_CHECK(expr) ::imgjit::cuda::detail::cu_check((expr), #expr, __FILE__, __LINE__)
#define NVRTC_CHECK(expr) ::imgjit::cuda::detail::nvrtc_check((expr), #expr, __FILE__, __LINE__)
#define CU_CHECK_NOTHROW(expr) \
  ::imgjit::cuda::detail::cu_check_nothrow((expr), #expr, __FILE__, __LINE__)

}  // namespace imgjit::cuda
