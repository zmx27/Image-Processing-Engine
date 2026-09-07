# Phase 2 (docs/PLAN.md): file in, file out, over the portable core. Builds and runs
# on a Mac with no CUDA toolkit present.
add_executable(imgjit-cli imgjit-cli.cpp)
target_link_libraries(imgjit-cli PRIVATE imgjit_cpu imgjit_io)

# The Phase 2 gate: the example chain from the plan, end to end on a real PNG.
add_test(
  NAME phase2_cli_grayscale_sobel
  COMMAND imgjit-cli --ops "grayscale,sobel"
          "${CMAKE_SOURCE_DIR}/tests/testdata/checkerboard_16x16_rgb.png"
          "${CMAKE_CURRENT_BINARY_DIR}/checkerboard_grayscale_sobel.png")

# Everything below needs an NVIDIA GPU, so it is Colab-only.
if(NOT IMGJIT_ENABLE_CUDA)
  return()
endif()

# Phase 1a: PTX produced ahead of time by nvcc, so the driver path can be proven
# with JIT out of the picture. Deliberately an add_custom_command rather than
# enable_language(CUDA): NVRTC compiles at runtime, so the project needs no CUDA
# toolchain, and enabling one invites <<<>>> and cudart linkage (CLAUDE.md
# invariant 8, docs/PLAN.md Phase 1). Generated rather than checked in so the blob
# cannot go stale against invert_aot.cu.
set(imgjit_aot_ptx "${CMAKE_CURRENT_BINARY_DIR}/invert_aot.ptx")
add_custom_command(
  OUTPUT "${imgjit_aot_ptx}"
  COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx
          -o "${imgjit_aot_ptx}" "${CMAKE_CURRENT_SOURCE_DIR}/invert_aot.cu"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/invert_aot.cu"
  COMMENT "nvcc --ptx invert_aot.cu (Phase 1a ahead-of-time blob)"
  VERBATIM)
add_custom_target(imgjit_aot_ptx_target DEPENDS "${imgjit_aot_ptx}")

add_executable(imgjit-spike imgjit-spike.cpp)
add_dependencies(imgjit-spike imgjit_aot_ptx_target)
target_link_libraries(imgjit-spike PRIVATE imgjit_cuda imgjit_stb)
target_compile_definitions(imgjit-spike PRIVATE IMGJIT_AOT_PTX_PATH="${imgjit_aot_ptx}")

# Both checkpoints run against a 3-channel and a 4-channel fixture; the spike exits
# non-zero unless every byte matches the scalar CPU inversion.
foreach(fixture checkerboard_16x16_rgb gradient_32x32_rgba)
  add_test(
    NAME phase1_spike_${fixture}
    COMMAND imgjit-spike "${CMAKE_SOURCE_DIR}/tests/testdata/${fixture}.png"
            --out "${CMAKE_CURRENT_BINARY_DIR}/${fixture}_inverted.png"
            --dump-ptx "${CMAKE_CURRENT_BINARY_DIR}/${fixture}_invert_jit.ptx")
endforeach()
