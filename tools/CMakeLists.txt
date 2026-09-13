# Phase 2/3 (docs/PLAN.md): file in, file out, over the real IBackend. Builds and runs
# on a Mac with no CUDA toolkit present; imgjit_codegen is linked unconditionally so
# `--dump-source` works there too.
add_executable(imgjit-cli imgjit-cli.cpp)
# imgjit_cpu is not listed here: imgjit_codegen already links it PUBLIC (codegen.h
# includes imgjit/core/kernel_key.h), so listing it again just relinks the same
# archive twice.
target_link_libraries(imgjit-cli PRIVATE imgjit_io imgjit_codegen)

# The Phase 2 gate: the example chain from the plan, end to end on a real PNG.
add_test(
  NAME phase2_cli_grayscale_sobel
  COMMAND imgjit-cli --ops "grayscale,sobel"
          "${CMAKE_SOURCE_DIR}/tests/testdata/checkerboard_16x16_rgb.png"
          "${CMAKE_CURRENT_BINARY_DIR}/checkerboard_grayscale_sobel.png")

# Phase 7, portable half: --tile from the flag through codegen to a dumped kernel, on a
# machine with no GPU. Whether that kernel computes the right thing is
# imgjit-gpu-tests [tiling]'s job.
add_test(
  NAME phase7_cli_dump_tiled_source
  COMMAND imgjit-cli --ops "grayscale,gaussian:1.4,sobel" --tile 16
          --dump-source "${CMAKE_CURRENT_BINARY_DIR}/phase7_tiled_chain.cu"
          "${CMAKE_SOURCE_DIR}/tests/testdata/checkerboard_16x16_rgb.png"
          "${CMAKE_CURRENT_BINARY_DIR}/checkerboard_phase7_cpu.png")

# Phase 4 (docs/PLAN.md, Track N): the server and the reference client. Both are
# portable — the whole network layer is proven against the CPU backend on a Mac, so a
# bug found here is never confused with a CUDA bug. Phase 5 adds `--backend cuda`,
# which changes only which backend the factory returns.
add_executable(imgjit-server imgjit-server.cpp)
target_link_libraries(imgjit-server PRIVATE imgjit_net imgjit_cpu)

add_executable(imgjit-client imgjit-client.cpp)
target_link_libraries(imgjit-client PRIVATE imgjit_net imgjit_io)

# The tools' own gate: ephemeral port handshake, a pipelining client asking for both
# result paths, clean shutdown on SIGINT. Correctness against the oracle and every
# malformed-frame case are asserted by imgjit-tests [server].
add_test(
  NAME phase4_server_client_smoke
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/phase4_smoke.sh"
          "$<TARGET_FILE:imgjit-server>" "$<TARGET_FILE:imgjit-client>"
          "${CMAKE_SOURCE_DIR}/tests/testdata/checkerboard_16x16_rgb.png"
          "${CMAKE_CURRENT_BINARY_DIR}/phase4_smoke")

# Everything below needs an NVIDIA GPU, so it is Colab-only.
if(NOT IMGJIT_ENABLE_CUDA)
  return()
endif()

target_link_libraries(imgjit-cli PRIVATE imgjit_cuda)
target_compile_definitions(imgjit-cli PRIVATE IMGJIT_ENABLE_CUDA)

# Phase 5: the same swap for the server. imgjit_net stays where it is — the network
# layer links no CUDA in either build, which is what keeps invariant 1's grep silent
# over src/net/.
target_link_libraries(imgjit-server PRIVATE imgjit_cuda)
target_compile_definitions(imgjit-server PRIVATE IMGJIT_ENABLE_CUDA)

# The Phase 3 gate's CLI half: the same chain on the GPU, dumping both the generated
# source and the PTX, and repeating so the cold-compile / warm-hit split is visible in
# the ctest log. Correctness against the CPU oracle is asserted by imgjit-gpu-tests.
add_test(
  NAME phase3_cli_cuda_grayscale_gaussian_sobel
  COMMAND imgjit-cli --backend cuda --ops "grayscale,gaussian:1.4,sobel" --repeat 3
          --dump-source "${CMAKE_CURRENT_BINARY_DIR}/phase3_chain.cu"
          --dump-ptx "${CMAKE_CURRENT_BINARY_DIR}/phase3_chain.ptx"
          "${CMAKE_SOURCE_DIR}/tests/testdata/checkerboard_16x16_rgb.png"
          "${CMAKE_CURRENT_BINARY_DIR}/checkerboard_phase3.png")

# Phase 7: the same chain through the tiled variant on the GPU, dumping the source and PTX
# that were actually compiled — the __shared__ tile is the thing to read in them.
add_test(
  NAME phase7_cli_cuda_tiled
  COMMAND imgjit-cli --backend cuda --tile 16 --ops "grayscale,gaussian:1.4,sobel" --repeat 3
          --dump-source "${CMAKE_CURRENT_BINARY_DIR}/phase7_tiled_chain_cuda.cu"
          --dump-ptx "${CMAKE_CURRENT_BINARY_DIR}/phase7_tiled_chain.ptx"
          "${CMAKE_SOURCE_DIR}/tests/testdata/checkerboard_16x16_rgb.png"
          "${CMAKE_CURRENT_BINARY_DIR}/checkerboard_phase7.png")

# Phase 1a: PTX produced ahead of time by nvcc, so the driver path can be proven
# with JIT out of the picture. Deliberately an add_custom_command rather than
# enable_language(CUDA): NVRTC compiles at runtime, so the project needs no CUDA
# toolchain, and enabling one invites <<<>>> and cudart linkage (invariant 8,
# docs/PLAN.md Phase 1). Generated rather than checked in so the blob
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
