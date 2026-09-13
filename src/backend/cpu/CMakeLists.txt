# The scalar CPU backend — invariant 9: the correctness oracle every GPU
# path is diffed against, and the backend the whole network layer is tested on before
# any CUDA code exists (docs/PLAN.md, Track N).
add_library(imgjit_cpu STATIC
  cpu_backend.cpp
  ops.cpp)
target_link_libraries(imgjit_cpu PUBLIC imgjit_core)
