# Phase 5's timing harness (docs/PLAN.md). Phase 8 widens the same binary over the
# orthogonal matrix rather than starting a second one, so the numbers Phases 6 and 7 are
# gated against stay like-for-like.
#
# Portable, and links no CUDA on purpose: it drives the server over a socket, so the
# backend on the other end is the variable under measurement rather than a build option
# here. That is also what lets a CPU-backend baseline be recorded on a Mac.
add_executable(imgjit-bench imgjit-bench.cpp)
target_link_libraries(imgjit-bench PRIVATE imgjit_net)
