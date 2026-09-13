// Phase 1a fixture (docs/PLAN.md): compiled to PTX ahead of time by `nvcc --ptx`
// via an add_custom_command in tools/CMakeLists.txt, so checkpoint 1a proves the
// driver plumbing with JIT out of the picture.
//
// This file is never compiled by the project's C++ toolchain and CMake's CUDA
// language is deliberately not enabled — see docs/PLAN.md Phase 1 and invariant 8.
//
// Phase 1b compiles a byte-for-byte equivalent kernel through NVRTC instead; the
// twin lives in kInvertSource in imgjit-spike.cpp. Keep the two in sync.

extern "C" __global__ void invert_u8(unsigned char* data, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    data[i] = (unsigned char)(255 - data[i]);
  }
}
