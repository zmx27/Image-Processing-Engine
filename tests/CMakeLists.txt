# Catch2 is vendored as the v3 amalgamation, which supplies main(). Warnings are
# disabled for it only: it is third-party code and the project builds -Werror.
add_library(catch2 STATIC "${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp")
target_include_directories(catch2 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/catch2")
set_source_files_properties("${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp"
  PROPERTIES COMPILE_OPTIONS "-w")

# The portable suite. Everything here builds on macOS with no CUDA toolkit — including
# the codegen tests, which assert the *text* of the generated kernel and so need no GPU.
add_executable(imgjit-tests
  test_codegen.cpp
  test_cpu_backend.cpp
  test_cpu_ops.cpp
  test_image_io.cpp
  test_kernel_key.cpp
  test_op_chain.cpp
  test_protocol.cpp
  test_server.cpp
  test_slot_pool.cpp)
# imgjit_io is not listed: imgjit_net already pulls it in for the server's flags-bit-1
# write path, and naming the same archive twice is a duplicate-library link warning.
target_link_libraries(imgjit-tests PRIVATE catch2 imgjit_codegen imgjit_net)
target_compile_definitions(imgjit-tests PRIVATE
  IMGJIT_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata"
  IMGJIT_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}")

# Registered per Catch2 tag rather than as one opaque target, so a ctest failure names
# the layer that broke. [net] is the codec and the concurrency primitives, which are
# pure functions and threads with no sockets; [server] is the end-to-end suite, and it
# is the one that has to be clean under BOTH sanitizer builds (docs/PLAN.md Phase 4).
foreach(tag core cpu io codegen net server)
  add_test(NAME imgjit_unit_${tag} COMMAND imgjit-tests "[${tag}]")
endforeach()

if(NOT IMGJIT_ENABLE_CUDA)
  return()
endif()

# A SEPARATE binary, not another tag on imgjit-tests: the portable suite must keep
# building and running on a machine with no toolkit, and linking the driver into it
# would end that. This one carries the Phase 3 gate — every chain in the corpus against
# the CPU oracle, plus the compile-counter assertions that make "compiles exactly once"
# a claim rather than a hope — the Phase 5 gate, which is the same oracle diff with the
# whole server and several concurrent connections in between, and the Phase 6 gate:
# genuine multi-stream residency, checksum stability under sustained load, and context
# recreation.
add_executable(imgjit-gpu-tests
  test_cuda_backend.cpp
  test_gpu_faults.cpp
  test_gpu_pipeline.cpp
  test_gpu_server.cpp)
# imgjit_cpu is not listed here: imgjit_cuda -> imgjit_codegen -> imgjit_cpu is all
# PUBLIC, so cpu/ops.h (used directly for the oracle diff) is already on the include
# path and imgjit_cpu is already in the link. imgjit_net IS listed: Phase 5 puts the
# real server in the test, and the network layer links no CUDA in either build.
target_link_libraries(imgjit-gpu-tests PRIVATE catch2 imgjit_cuda imgjit_net)
target_compile_definitions(imgjit-gpu-tests PRIVATE
  IMGJIT_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}")

add_test(NAME phase3_gpu_oracle_diff COMMAND imgjit-gpu-tests "[oracle]")
add_test(NAME phase3_gpu_kernel_cache COMMAND imgjit-gpu-tests "[cache]")
add_test(NAME phase5_gpu_server COMMAND imgjit-gpu-tests "[gpu-server]")

# Phase 6, split three ways for the same reason as above — "the pipeline is broken" and
# "recovery is broken" are different mornings.
add_test(NAME phase6_gpu_async COMMAND imgjit-gpu-tests "[async]")
add_test(NAME phase6_gpu_stress COMMAND imgjit-gpu-tests "[stress]")
add_test(NAME phase6_gpu_recovery COMMAND imgjit-gpu-tests "[recovery]")

# Phase 7: the tiled variant against the CPU oracle and against the naive kernel, at
# several tile sizes, plus the tile half of the kernel identity.
add_test(NAME phase7_gpu_tiling COMMAND imgjit-gpu-tests "[tiling]")

# Phase 8: parameterized constants against the CPU oracle and the baked kernel, plus
# the one-compile-per-shape claim that taking the values out of the key makes.
add_test(NAME phase8_gpu_constants COMMAND imgjit-gpu-tests "[constants]")

# Phase 8's error-injection pass, kept apart from [recovery] because it asserts something
# stronger: a REAL sticky driver error (a null-pointer store from a kernel), not a call to
# recreate_context(), and malformed protocol frames against a GPU-backed server rather
# than the CPU one. "The server survives" is the claim under test, so a failure here
# should never be confused with a pipeline bug.
#
# THREE CASES, THREE PROCESSES, AND THAT IS NOT OPTIONAL. An illegal access poisons the
# whole PROCESS, not just the context that caused it — measured on a T4, cuCtxCreate
# afterwards returns CUDA_ERROR_ILLEGAL_ADDRESS, so anything sharing the process with an
# injecting case dies in the CudaBackend constructor. The first version of this file ran
# all three under one tag and the two innocent cases failed exactly that way. ctest runs
# each add_test as its own process, which is what keeps them isolated.
add_test(NAME phase8_gpu_faults COMMAND imgjit-gpu-tests "[faults]")
add_test(NAME phase8_gpu_fault_backend COMMAND imgjit-gpu-tests "[fault-backend]")
add_test(NAME phase8_gpu_fault_server COMMAND imgjit-gpu-tests "[fault-server]")
