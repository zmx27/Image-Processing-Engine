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
# would end that. This one is the Phase 3 gate — every chain in the corpus against the
# CPU oracle, plus the compile-counter assertions that make "compiles exactly once"
# a claim rather than a hope.
add_executable(imgjit-gpu-tests test_cuda_backend.cpp)
# imgjit_cpu is not listed here: imgjit_cuda -> imgjit_codegen -> imgjit_cpu is all
# PUBLIC, so cpu/ops.h (used directly for the oracle diff) is already on the include
# path and imgjit_cpu is already in the link.
target_link_libraries(imgjit-gpu-tests PRIVATE catch2 imgjit_cuda)

add_test(NAME phase3_gpu_oracle_diff COMMAND imgjit-gpu-tests "[oracle]")
add_test(NAME phase3_gpu_kernel_cache COMMAND imgjit-gpu-tests "[cache]")
