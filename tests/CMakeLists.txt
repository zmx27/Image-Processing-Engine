# Catch2 is vendored as the v3 amalgamation, which supplies main(). Warnings are
# disabled for it only: it is third-party code and the project builds -Werror.
add_library(catch2 STATIC "${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp")
target_include_directories(catch2 PUBLIC "${CMAKE_SOURCE_DIR}/third_party/catch2")
set_source_files_properties("${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp"
  PROPERTIES COMPILE_OPTIONS "-w")

add_executable(imgjit-tests
  test_cpu_backend.cpp
  test_cpu_ops.cpp
  test_image_io.cpp
  test_kernel_key.cpp
  test_op_chain.cpp)
target_link_libraries(imgjit-tests PRIVATE catch2 imgjit_cpu imgjit_io)
target_compile_definitions(imgjit-tests PRIVATE
  IMGJIT_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata"
  IMGJIT_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}")

# Registered per Catch2 tag rather than as one opaque target, so a ctest failure names
# the layer that broke.
foreach(tag core cpu io)
  add_test(NAME imgjit_unit_${tag} COMMAND imgjit-tests "[${tag}]")
endforeach()
