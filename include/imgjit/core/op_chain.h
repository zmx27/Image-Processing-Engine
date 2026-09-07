#pragma once

// The op-chain IR: what a client asks for, and the sole input to both the CPU oracle
// and (from Phase 3) CUDA codegen.
//
// The op set is CLOSED AT SIX (docs/PLAN.md Phase 2, ARCHITECTURE.md decision 5):
// pointwise `grayscale`, `invert`, `brightness`, `threshold`; stencil `gaussian`,
// `sobel`. Adding ops is the project's named #1 over-scoping risk; the cap is what
// keeps the fusion story finite and the test corpus complete.
//
// Semantics every implementation of these ops must share, because the GPU is diffed
// against the CPU oracle within 1 LSB and a definitional difference reads as a bug:
//
//   * Channel count is invariant across a chain. `grayscale` writes the luminance
//     into R, G and B rather than reducing to one channel — an op that changed the
//     layout would force every fusion stage in Phase 3 to renegotiate it.
//   * Alpha (channel 3 of a 4-channel image) passes through every op untouched.
//     Blurring or inverting opacity is never what the caller meant.
//   * Float ops run on samples normalized to [0, 1]; results are clamped and rounded
//     back to uint8 on store.
//   * Stencils use clamped (replicate) edge addressing.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace imgjit {

enum class OpKind : std::uint8_t {
  kGrayscale = 0,
  kInvert = 1,
  kBrightness = 2,
  kThreshold = 3,
  kGaussian = 4,
  kSobel = 5,
};

struct Op {
  OpKind kind{};
  // Canonical (quantized) parameter; 0 for the ops that take none.
  float param{0.0F};

  friend bool operator==(const Op&, const Op&) = default;
};

struct OpChain {
  std::vector<Op> ops;

  friend bool operator==(const OpChain&, const OpChain&) = default;
};

// Static description of one op. The parser, the canonicalizer and (later) codegen all
// read this table rather than each carrying their own copy of the names and bounds.
struct OpSpec {
  OpKind kind{};
  std::string_view name;
  bool takes_param{false};
  bool is_stencil{false};
  float default_param{0.0F};
  float min_param{0.0F};
  float max_param{0.0F};
};

// Parameter bounds are not decoration. `gaussian`'s cap bounds the stencil radius
// (radius = ceil(3 * sigma) <= 12), which Phase 3 bakes into an unrolled loop and
// Phase 7 turns into a shared-memory halo — an unbounded sigma is an unbounded
// kernel. Defaults exist so "implicit defaults made explicit" is something the
// canonicalizer can actually do.
inline constexpr std::array<OpSpec, 6> kOpSpecs{{
    {OpKind::kGrayscale, "grayscale", false, false, 0.0F, 0.0F, 0.0F},
    {OpKind::kInvert, "invert", false, false, 0.0F, 0.0F, 0.0F},
    {OpKind::kBrightness, "brightness", true, false, 0.0F, -1.0F, 1.0F},
    {OpKind::kThreshold, "threshold", true, false, 0.5F, 0.0F, 1.0F},
    {OpKind::kGaussian, "gaussian", true, true, 1.0F, 0.1F, 4.0F},
    {OpKind::kSobel, "sobel", false, true, 0.0F, 0.0F, 0.0F},
}};

const OpSpec& op_spec(OpKind kind);

// Parameters are quantized to two decimals here, at parse time, rather than only when
// hashing. Both would collapse `gaussian:1.4` and `gaussian:1.4000001` onto one cache
// entry; quantizing early additionally makes the op that runs identical to the one the
// cached kernel baked, instead of merely colliding with it (docs/PLAN.md Phase 2).
constexpr int kParamDecimals = 2;
float quantize_param(float value);

// Parses `"grayscale,gaussian:1.4,sobel,threshold:0.3"`. Whitespace around op names,
// colons and parameters is ignored; an omitted parameter takes the op's default.
// Returns nullopt on any malformed or out-of-range input, writing a human-readable
// reason to `error` when it is non-null. The returned chain is already canonical.
//
// An empty chain (`""`) is valid and means "no ops" — an identity pass.
std::optional<OpChain> parse_op_chain(std::string_view text, std::string* error = nullptr);

// The canonical text form: no whitespace, every parameter explicit, floats printed
// from their quantized value with trailing zeros stripped (`gaussian:1.40` ->
// `gaussian:1.4`). Op ORDER IS NEVER TOUCHED — `gaussian,sobel` and `sobel,gaussian`
// are different images, so canonicalization is parameter normalization only.
std::string canonical_string(const OpChain& chain);

}  // namespace imgjit
