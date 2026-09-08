#pragma once

// The stable half of every generated program: device helpers that do not vary with
// the op chain, held as ONE raw-string literal.
//
// There is deliberately no `kernels/` directory. NVRTC has no default include path,
// so generated source cannot #include anything at all (and must not reach for
// <cuda_runtime.h> — that is CLAUDE.md invariant 8). Keeping these ~40 lines here
// rather than in .cu files on disk avoids a runtime path dependency or a CMake embed
// step, and avoids leaving behind fragments no compiler ever checks. The readability
// need is served from the other end, by `imgjit-cli --dump-source`: what you read when
// debugging is the *generated* kernel, not the pieces it was assembled from
// (docs/ARCHITECTURE.md).
//
// Everything that varies — baked weights, radii, thresholds, the channel count, the
// stage structure — is emitted by codegen.cpp instead. The split is exactly "stable
// text here, every codegen input there".
//
// Portable: this is a string, not CUDA. It compiles on macOS with no toolkit present.

namespace imgjit::cuda {

// Two properties below are load-bearing for the <=1 LSB oracle diff, and both look
// like pointless pedantry until they are wrong:
//
//   * imgjit_load_sample DIVIDES by 255 rather than multiplying by 1/255f. The CPU
//     oracle divides, and 1/255 is not representable in binary floating point, so the
//     reciprocal form differs from it by an ulp on some inputs.
//   * imgjit_store_sample rounds with roundf, which is round-half-away-from-zero —
//     the rule std::lround follows in the oracle. __float2int_rn would round halves to
//     even instead and disagree on every exact .5.
inline constexpr const char* kKernelPrelude = R"CUDA(
// ---------------------------------------------------------------------------
// imgjit kernel prelude — stable device helpers (src/backend/cuda/kernel_prelude.h)
// Self-contained by necessity: NVRTC has no default include path.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int imgjit_clamp_coord(int v, int limit) {
  return v < 0 ? 0 : (v > limit - 1 ? limit - 1 : v);
}

__device__ __forceinline__ float imgjit_clamp01(float v) {
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__device__ __forceinline__ float imgjit_load_sample(unsigned char v) {
  return (float)v / 255.0f;
}

__device__ __forceinline__ unsigned char imgjit_store_sample(float v) {
  return (unsigned char)(int)roundf(imgjit_clamp01(v) * 255.0f);
}

// The fusion boundary, and the reason fusion is safe to compare against the oracle.
//
// The CPU oracle materializes a whole uint8 image between every pair of ops. A fused
// kernel keeps the value in a register instead, so unless it rounds here, fusing would
// silently change the NUMERICS rather than just the memory traffic — and the drift is
// not small: Sobel's coefficients sum to 8 in absolute value, so a half-LSB difference
// per tap amplifies to ~4 LSB, and a threshold that a sample straddles flips 0 <-> 255.
// Rounding here keeps fusion a pure memory optimization.
//
// Idempotent with imgjit_store_sample, so the call codegen emits before a store is
// redundant; it costs one roundf and buys a rule with no exceptions.
__device__ __forceinline__ float imgjit_quantize(float v) {
  return imgjit_load_sample(imgjit_store_sample(v));
}

// Rec. 601, in the oracle's association order. These three constants are the same
// numbers as imgjit::cpu::kLumaRed / kLumaGreen / kLumaBlue; a codegen test asserts
// they have not drifted apart.
__device__ __forceinline__ float imgjit_luma(float r, float g, float b) {
  return 0.299f * r + 0.587f * g + 0.114f * b;
}
)CUDA";

}  // namespace imgjit::cuda
