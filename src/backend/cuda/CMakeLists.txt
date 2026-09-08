# This is the only directory permitted to include <cuda.h> / <nvrtc.h> (CLAUDE.md
# invariant 1), so the include directory is src/ and consumers spell the path in full.
#
# It builds as TWO targets, because not everything here needs a GPU:
#
#   imgjit_codegen — Phase 3 codegen. Emits CUDA source as a string, links no CUDA,
#                    builds on macOS. Split out so the generated source is unit-tested
#                    locally instead of only on Colab, and so `imgjit-cli --dump-source`
#                    works without a toolkit.
#   imgjit_cuda    — the context, the kernel cache and the backend. Calls the driver,
#                    so it exists only when a toolkit was found.
add_library(imgjit_codegen STATIC codegen.cpp)
target_include_directories(imgjit_codegen PUBLIC "${CMAKE_SOURCE_DIR}/src")
# PUBLIC, not PRIVATE: codegen.h itself #includes imgjit/core/kernel_key.h, so anything
# that links imgjit_codegen (imgjit_cuda, imgjit-cli, the tests) needs imgjit_cpu's
# public include path transitively, not just codegen.cpp's own translation unit.
# Also not decoration: codegen bakes gaussian weights straight out of the CPU oracle so
# the two convolve with bit-identical numbers rather than two expf results that agree.
target_link_libraries(imgjit_codegen PUBLIC imgjit_cpu)

if(NOT IMGJIT_ENABLE_CUDA)
  return()
endif()

# CUDA::cudart is never linked — the Runtime API would create an implicit primary
# context that conflicts with the one the worker owns (CLAUDE.md invariant 8).
add_library(imgjit_cuda STATIC
  cuda_backend.cpp
  kernel_cache.cpp)
target_include_directories(imgjit_cuda PUBLIC "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(imgjit_cuda PUBLIC imgjit_codegen CUDA::cuda_driver CUDA::nvrtc)
