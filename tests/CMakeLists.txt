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
  test_op_chain.cpp)
target_link_libraries(imgjit-tests PRIVATE catch2 imgjit_cpu imgjit_io imgjit_codegen)
target_compile_definitions(imgjit-tests PRIVATE
  IMGJIT_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata"
  IMGJIT_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}")

# Registered per Catch2 tag rather than as one opaque target, so a ctest failure names
# the layer that broke.
foreach(tag core cpu io codegen)
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
# imgjit_cpu is named explicitly, not inherited: this binary diffs against the oracle,
# so it includes cpu/ops.h directly, and imgjit_codegen links that target privately.
target_link_libraries(imgjit-gpu-tests PRIVATE catch2 imgjit_cuda imgjit_cpu)

add_test(NAME phase3_gpu_oracle_diff COMMAND imgjit-gpu-tests "[oracle]")
add_test(NAME phase3_gpu_kernel_cache COMMAND imgjit-gpu-tests "[cache]")
