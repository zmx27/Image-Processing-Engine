// Phase 3 codegen (docs/PLAN.md). These assert the *text* of the generated kernel, so
// they run on a Mac with no CUDA toolkit — which is the point of splitting
// imgjit_codegen out of imgjit_cuda.
//
// What they cannot check is that the source compiles, or that it computes the right
// numbers. That is imgjit-gpu-tests' job on Colab, and no assertion here is a
// substitute for it. What they DO check is the structure a GPU test would find hard to
// attribute: the fusion plan, which constants got baked, and — the one that would
// otherwise only surface as a mystery benchmark number — that nothing resolution-
// dependent leaked into the source.

#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_set>

#include "backend/cuda/codegen.h"
#include "backend/cuda/kernel_prelude.h"
#include "catch_amalgamated.hpp"
#include "imgjit/backend/cpu/ops.h"
#include "imgjit/core/kernel_key.h"

using imgjit::ConstantsMode;
using imgjit::KernelKey;
using imgjit::parse_op_chain;
using imgjit::TileVariant;
using imgjit::cuda::emit_cuda_source;
using imgjit::cuda::GeneratedProgram;

namespace {

KernelKey key_for(std::string_view chain, int channels = 3) {
  const auto parsed = parse_op_chain(chain);
  REQUIRE(parsed.has_value());
  KernelKey key;
  key.chain = *parsed;
  key.channels = channels;
  return key;
}

GeneratedProgram emit(std::string_view chain, int channels = 3) {
  return emit_cuda_source(key_for(chain, channels));
}

bool contains(const std::string& haystack, std::string_view needle) {
  return haystack.find(needle) != std::string::npos;
}

std::size_t count_of(const std::string& haystack, std::string_view needle) {
  std::size_t total = 0;
  for (std::size_t at = haystack.find(needle); at != std::string::npos;
       at = haystack.find(needle, at + needle.size())) {
    ++total;
  }
  return total;
}

// The same %.9g round-trip codegen uses, so a test can look for the exact literal a
// float was printed as rather than an approximation of it.
std::string literal(float value) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(value));
  std::string printed(text);
  if (printed.find_first_of(".eE") == std::string::npos) {
    printed += ".0";
  }
  return printed + "f";
}

}  // namespace

TEST_CASE("an empty chain emits no kernel at all", "[codegen]") {
  // An identity pass. The backend short-circuits it rather than asking NVRTC for a
  // translation unit with no __global__ in it.
  CHECK(emit("").stages.empty());
}

TEST_CASE("a pointwise run is one kernel", "[codegen]") {
  const GeneratedProgram program = emit("grayscale,invert,brightness:0.2,threshold:0.4");
  REQUIRE(program.stages.size() == 1);
  CHECK_FALSE(program.stages.front().is_stencil);
  CHECK(count_of(program.source, "__global__") == 1);
}

TEST_CASE("stencils bound the stage count, pointwise ops never add to it", "[codegen]") {
  // docs/ARCHITECTURE.md decision 5: at most #stencil_ops + 1 kernels. The interleaved
  // chain is the one that would blow past that if pointwise runs each got their own
  // stage — five ops, two stencils, and still only two kernels.
  CHECK(emit("sobel").stages.size() == 1);
  CHECK(emit("grayscale,sobel").stages.size() == 1);
  CHECK(emit("sobel,invert").stages.size() == 1);
  CHECK(emit("gaussian:1.4,sobel").stages.size() == 2);
  CHECK(emit("invert,gaussian:1,invert,sobel,invert").stages.size() == 2);
  CHECK(emit("grayscale,gaussian:1.4,sobel,threshold:0.3").stages.size() == 2);

  for (const auto& stage : emit("gaussian:1.4,sobel").stages) {
    CHECK(stage.is_stencil);
  }
}

TEST_CASE("a pointwise run fuses into the neighbouring stencil, not its own kernel",
          "[codegen]") {
  // Fused forwards: grayscale has to run at every tap the sobel reads, so it lands in
  // the tap helper rather than in the kernel body.
  const GeneratedProgram forward = emit("grayscale,sobel");
  REQUIRE(forward.stages.size() == 1);
  const std::size_t tap_at = forward.source.find("void imgjit_stage0_tap(");
  const std::size_t kernel_at = forward.source.find("__global__ void imgjit_stage0(");
  REQUIRE(tap_at != std::string::npos);
  REQUIRE(kernel_at != std::string::npos);
  CHECK(forward.source.find("imgjit_luma", tap_at) < kernel_at);

  // Fused backwards, which is the cheaper direction and therefore the preferred one:
  // an epilogue runs once per output pixel, a prologue once per tap. So the invert
  // here must be in the kernel body, after the stencil — not in the tap helper.
  const GeneratedProgram backward = emit("sobel,invert");
  REQUIRE(backward.stages.size() == 1);
  const std::size_t back_kernel_at = backward.source.find("__global__ void imgjit_stage0(");
  REQUIRE(back_kernel_at != std::string::npos);
  CHECK(backward.source.find("1.0f - v[0]") > back_kernel_at);

  // Only the first stage can carry a prologue, since every later pointwise run has a
  // preceding stencil to attach to. One tap helper does the folding for the chain.
  CHECK(count_of(emit("grayscale,gaussian:1.4,invert,sobel,invert").source, "_tap(\n") == 2);
}

TEST_CASE("dimensions are launch arguments and never reach the source", "[codegen]") {
  // CLAUDE.md invariant 4, checked from the codegen end. KernelKey has no dimension
  // field to leak, so this asserts the other half: the kernel takes them as
  // parameters, which is what makes one compile serve every resolution.
  const GeneratedProgram program = emit("gaussian:1.4,sobel");
  CHECK(contains(program.source, "int width, int height"));
  CHECK(count_of(program.source, "int width, int height") == 4);  // 2 taps + 2 kernels
}

TEST_CASE("stencil constants are baked from the CPU oracle's own numbers", "[codegen]") {
  // Not "some weights that look gaussian": the literals must be the exact floats the
  // oracle convolves with, or the two run different filters and the <=1 LSB diff is
  // measuring the wrong thing.
  const float sigma = 1.4F;
  const GeneratedProgram program = emit("gaussian:1.4");
  const std::vector<float> weights = imgjit::cpu::gaussian_weights_1d(sigma);

  REQUIRE(weights.size() == 11);  // radius = ceil(3 * 1.4) = 5
  CHECK(contains(program.source, "_w[11] = {"));
  for (const float weight : weights) {
    CHECK(contains(program.source, literal(weight)));
  }

  // The radius is baked as a loop bound, so the stencil footprint is a compile-time
  // constant (which is what Phase 7's __shared__ tile will need).
  CHECK(contains(program.source, "for (int dy = -5; dy <= 5; ++dy)"));
}

TEST_CASE("pointwise parameters are baked as literals", "[codegen]") {
  CHECK(contains(emit("brightness:0.25").source, literal(0.25F)));
  CHECK(contains(emit("threshold:0.3").source, literal(0.3F)));
}

TEST_CASE("every op boundary re-quantizes to uint8 precision", "[codegen]") {
  // The property that makes fusion safe to diff against the oracle at <=1 LSB: the
  // oracle stores a uint8 image between ops, so a fused kernel must round at the same
  // points or it is running a different filter.
  //
  // One call per fused op, at 1 channel where grayscale is a no-op: invert, brightness,
  // threshold. The +1 is the prelude's own definition of the helper, which the same
  // substring matches.
  const GeneratedProgram program = emit("grayscale,invert,brightness:0.1,threshold:0.5", 1);
  CHECK(count_of(program.source, "imgjit_quantize(") == 3 + 1);

  // At 3 channels the same chain rounds per colour channel, and grayscale is no longer
  // a no-op: 4 ops x 3 channels, except grayscale which quantizes the luma once.
  const GeneratedProgram wide = emit("grayscale,invert,brightness:0.1,threshold:0.5", 3);
  CHECK(count_of(wide.source, "imgjit_quantize(") == 1 + (3 * 3) + 1);

  const GeneratedProgram fused = emit("grayscale,sobel");
  const std::size_t tap_at = fused.source.find("void imgjit_stage0_tap(");
  const std::size_t kernel_at = fused.source.find("__global__ void");
  CHECK(fused.source.find("imgjit_quantize(imgjit_luma", tap_at) < kernel_at);
  CHECK(contains(fused.source, "imgjit_quantize(sqrtf("));
}

TEST_CASE("the channel count changes indexing and alpha handling", "[codegen]") {
  CHECK(contains(emit("invert", 4).source, "dst[i + 3] = src[i + 3];"));
  CHECK_FALSE(contains(emit("invert", 3).source, "dst[i + 3] = src[i + 3];"));
  CHECK_FALSE(contains(emit("invert", 1).source, "dst[i + 3] = src[i + 3];"));

  // Baked as a literal, so the stride multiply is a shift and not a load.
  CHECK(contains(emit("invert", 4).source, "* 4;"));
  CHECK(contains(emit("invert", 3).source, "* 3;"));

  // A 1-channel sample is already luminance; the oracle leaves it bit-identical rather
  // than pushing it through a float round trip, and so must the kernel. Matched on the
  // CALL, not the name — the prelude always defines imgjit_luma.
  CHECK_FALSE(contains(emit("grayscale", 1).source, "imgjit_luma(v["));
  CHECK(contains(emit("grayscale", 3).source, "imgjit_luma(v["));
}

TEST_CASE("kernel names are unique and every declared stage exists in the source",
          "[codegen]") {
  const GeneratedProgram program = emit("grayscale,gaussian:1.4,invert,sobel,threshold:0.2", 4);
  std::unordered_set<std::string> names;
  for (const auto& stage : program.stages) {
    CHECK(names.insert(stage.kernel_name).second);
    CHECK(contains(program.source, "__global__ void " + stage.kernel_name + "("));
  }
  CHECK(count_of(program.source, "__global__") == program.stages.size());
}

TEST_CASE("canonically equivalent chains generate byte-identical source", "[codegen]") {
  // The other half of the cache's correctness: these spellings share one KernelKey, so
  // if they generated different source, one of them would be silently running the
  // other's kernel.
  const std::string expected = emit("gaussian:1.4").source;
  CHECK(emit("gaussian:1.40").source == expected);
  CHECK(emit(" gaussian : 1.4 ").source == expected);
  CHECK(emit("gaussian:1.4000001").source == expected);

  // And the converse: anything the key distinguishes must produce different source.
  CHECK(emit("gaussian:1.5").source != expected);
  CHECK(emit("gaussian:1.4", 4).source != expected);
  CHECK(emit("sobel,gaussian:1.4").source != emit("gaussian:1.4,sobel").source);
}

TEST_CASE("the prelude's luma constants have not drifted from the oracle's", "[codegen]") {
  // kernel_prelude.h holds these as stable text while imgjit::cpu holds them as
  // constants. Two copies of a number that must agree is exactly the drift this
  // catches — and it would otherwise show up as a GPU/CPU mismatch on Colab, hours
  // away from the edit that caused it.
  //
  // Pinned from both sides, because one side alone proves nothing: the string checks
  // fix what the kernel computes, and the value checks fix what the oracle computes.
  // Changing either constant without the other fails here.
  const std::string prelude = imgjit::cuda::kKernelPrelude;
  CHECK(contains(prelude, "0.299f * r"));
  CHECK(contains(prelude, "0.587f * g"));
  CHECK(contains(prelude, "0.114f * b"));
  CHECK(imgjit::cpu::kLumaRed == 0.299F);
  CHECK(imgjit::cpu::kLumaGreen == 0.587F);
  CHECK(imgjit::cpu::kLumaBlue == 0.114F);
}

TEST_CASE("variants reserved for later phases are refused, not silently ignored",
          "[codegen]") {
  // A key field codegen ignores is worse than one it rejects: the cache would hand
  // back the naive kernel for a tiled request and the benchmark would report a
  // speedup of exactly zero, with nothing to show why.
  KernelKey tiled = key_for("sobel");
  tiled.tile = TileVariant::kTiled;
  tiled.tile_size = 16;
  CHECK_THROWS_AS(emit_cuda_source(tiled), std::invalid_argument);

  KernelKey parameterized = key_for("sobel");
  parameterized.constants = ConstantsMode::kParameterized;
  CHECK_THROWS_AS(emit_cuda_source(parameterized), std::invalid_argument);

  KernelKey two_channel = key_for("sobel");
  two_channel.channels = 2;
  CHECK_THROWS_AS(emit_cuda_source(two_channel), std::invalid_argument);
}
