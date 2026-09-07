# Header-only for Phase 1: CU_CHECK / NVRTC_CHECK and the RAII wrappers. This is the
# only directory permitted to include <cuda.h> / <nvrtc.h> (CLAUDE.md invariant 1),
# so the include directory is src/ and consumers spell the path in full.
#
# CUDA::cudart is never linked — the Runtime API would create an implicit primary
# context that conflicts with the one the worker owns (CLAUDE.md invariant 8).
add_library(imgjit_cuda INTERFACE)
target_include_directories(imgjit_cuda INTERFACE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(imgjit_cuda INTERFACE CUDA::cuda_driver CUDA::nvrtc)
