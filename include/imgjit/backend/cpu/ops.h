#pragma once

// The scalar CPU implementation of the closed op set — CLAUDE.md invariant 9: this is
// the correctness oracle every GPU path is diffed against. Clarity beats speed here
// without exception; nothing in this file should ever be optimized.
//
// The exact semantics these functions define (channel invariance, alpha passthrough,
// normalized float math, clamped edge addressing) are spelled out in
// imgjit/core/op_chain.h and are binding on Phase 3's codegen.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <vector>

#include "imgjit/core/image.h"
#include "imgjit/core/op_chain.h"

namespace imgjit::cpu {

// Rec. 601 luma, matching what the generated kernel's prelude will compute.
inline constexpr float kLumaRed = 0.299F;
inline constexpr float kLumaGreen = 0.587F;
inline constexpr float kLumaBlue = 0.114F;

// Number of colour channels an op touches: everything but the alpha of an RGBA image.
constexpr int color_channels(int channels) { return channels == 4 ? 3 : channels; }

// Gaussian radius. Three sigma captures ~99.7% of the kernel's mass; the [0.1, 4.0]
// sigma bound in op_chain.h is what keeps this <= 12.
int gaussian_radius(float sigma);

// Normalized (2r+1)-tap 1D weights. Exposed because Phase 3 bakes these exact numbers
// into the generated source as literals, and the tests check the 2D outer product
// they imply sums to one.
std::vector<float> gaussian_weights_1d(float sigma);

Image apply_op(const Image& input, const Op& op);

// Folds the chain left to right. Order is semantically load-bearing and is never
// reordered anywhere in this project.
Image apply_chain(const Image& input, const OpChain& chain);

}  // namespace imgjit::cpu
