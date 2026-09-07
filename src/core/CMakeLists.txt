# The portable core: Image, the OpChain IR + parser + canonicalizer, and the kernel
# cache key. No CUDA and no sockets — this is what makes local Mac development
# possible at all (CLAUDE.md conventions).
add_library(imgjit_core STATIC
  image.cpp
  kernel_key.cpp
  op_chain.cpp)
target_include_directories(imgjit_core PUBLIC "${CMAKE_SOURCE_DIR}/include")
