# Portable subdirectories build everywhere, including macOS with no CUDA toolkit
# present.
add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(net)
add_subdirectory(backend/cpu)

# backend/cuda is added unconditionally and does its own gating: codegen emits CUDA
# *source text* and needs no toolkit, so it builds and is unit-tested on macOS, while
# everything in that directory which actually calls the driver stays behind
# IMGJIT_ENABLE_CUDA. See src/backend/cuda/CMakeLists.txt.
add_subdirectory(backend/cuda)
