// The scalar CPU ops (docs/PLAN.md Phase 2) — CLAUDE.md invariant 9's oracle, which
// means these tests are what the GPU's correctness ultimately rests on.
//
// Assertions here are analytic or property-based, never golden output files
// regenerated from this same code: a golden PNG produced by the implementation under
// test only proves it has not changed, and would have locked in a wrong Rec. 601
// coefficient or an off-by-one halo just as happily as a right one. What is checked
// instead: hand-computed values, invariants the filters must satisfy by definition
// (a normalized blur leaves a constant image constant; a gradient operator is zero on
// one), and a deliberately independent second implementation for the Gaussian.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "catch_amalgamated.hpp"
#include "imgjit/backend/cpu/ops.h"
#include "imgjit/core/op_chain.h"
#include "imgjit/util/image_io.h"

using imgjit::Image;
using imgjit::Op;
using imgjit::OpKind;
using imgjit::cpu::apply_chain;
using imgjit::cpu::apply_op;

namespace {

Image fixture(const std::string& name) {
  return imgjit::load_png(std::string(IMGJIT_TESTDATA_DIR) + "/" + name);
}

Image image_from(int width, int height, int channels, const std::vector<std::uint8_t>& samples) {
  REQUIRE(samples.size() ==
          static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
              static_cast<std::size_t>(channels));
  return Image::from_bytes(width, height, channels, samples.data(),
                           static_cast<std::size_t>(width) * static_cast<std::size_t>(channels));
}

Image chain_of(const Image& input, std::string_view text) {
  const auto chain = imgjit::parse_op_chain(text);
  REQUIRE(chain.has_value());
  return apply_chain(input, *chain);
}

// The project's comparison bar for float ops: <= 1 LSB on uint8 output.
void require_within_one_lsb(const Image& actual, const Image& expected) {
  REQUIRE(actual.width() == expected.width());
  REQUIRE(actual.height() == expected.height());
  REQUIRE(actual.channels() == expected.channels());
  int worst = 0;
  for (int y = 0; y < actual.height(); ++y) {
    for (int x = 0; x < actual.width(); ++x) {
      for (int c = 0; c < actual.channels(); ++c) {
        worst = std::max(worst, std::abs(static_cast<int>(actual.at(x, y, c)) -
                                         static_cast<int>(expected.at(x, y, c))));
      }
    }
  }
  CHECK(worst <= 1);
}

// A deliberately different factorization of the same filter: two 1D passes instead of
// one 2D stencil. Clamping in x and in y are independent, so the separable form is
// mathematically identical to the outer-product form the oracle evaluates — any
// disagreement beyond float noise is a real bug in one of them, not in the method.
Image separable_gaussian_reference(const Image& input, float sigma) {
  const std::vector<float> weights = imgjit::cpu::gaussian_weights_1d(sigma);
  const int radius = imgjit::cpu::gaussian_radius(sigma);
  const int colors = imgjit::cpu::color_channels(input.channels());
  const auto clamp_to = [](int v, int limit) { return std::clamp(v, 0, limit - 1); };

  std::vector<float> horizontal(static_cast<std::size_t>(input.width()) *
                                static_cast<std::size_t>(input.height()) *
                                static_cast<std::size_t>(colors));
  const auto index = [&](int x, int y, int c) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(input.width()) +
            static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(colors) +
           static_cast<std::size_t>(c);
  };

  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      for (int c = 0; c < colors; ++c) {
        float sum = 0.0F;
        for (int d = -radius; d <= radius; ++d) {
          sum += weights[static_cast<std::size_t>(d + radius)] *
                 (static_cast<float>(input.at(clamp_to(x + d, input.width()), y, c)) / 255.0F);
        }
        horizontal[index(x, y, c)] = sum;
      }
    }
  }

  Image output(input.width(), input.height(), input.channels());
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      for (int c = 0; c < colors; ++c) {
        float sum = 0.0F;
        for (int d = -radius; d <= radius; ++d) {
          sum += weights[static_cast<std::size_t>(d + radius)] *
                 horizontal[index(x, clamp_to(y + d, input.height()), c)];
        }
        output.at(x, y, c) = static_cast<std::uint8_t>(
            std::lround(std::clamp(sum, 0.0F, 1.0F) * 255.0F));
      }
    }
  }
  if (input.channels() == 4) {
    for (int y = 0; y < input.height(); ++y) {
      for (int x = 0; x < input.width(); ++x) {
        output.at(x, y, 3) = input.at(x, y, 3);
      }
    }
  }
  return output;
}

}  // namespace

TEST_CASE("invert is exact on colour channels", "[cpu]") {
  // Integer pointwise op: the one place bit-equality is the right bar.
  const Image input = fixture("solid_4x4_rgb.png");
  const Image inverted = apply_op(input, Op{OpKind::kInvert, 0.0F});
  CHECK(inverted.at(0, 0, 0) == 255 - 200);
  CHECK(inverted.at(0, 0, 1) == 255 - 40);
  CHECK(inverted.at(0, 0, 2) == 255 - 40);
  CHECK(apply_op(inverted, Op{OpKind::kInvert, 0.0F}) == input);
}

TEST_CASE("grayscale writes Rec. 601 luma into every colour channel", "[cpu]") {
  const Image input = fixture("solid_4x4_rgb.png");  // every pixel is (200, 40, 40)
  const Image gray = apply_op(input, Op{OpKind::kGrayscale, 0.0F});

  // 0.299*200 + 0.587*40 + 0.114*40 = 87.84 -> 88
  CHECK(gray.channels() == 3);  // channel count is invariant across a chain
  for (int c = 0; c < 3; ++c) {
    CHECK(static_cast<int>(gray.at(0, 0, c)) == 88);
  }
}

TEST_CASE("grayscale leaves a 1-channel image bit-identical", "[cpu]") {
  // Already luminance; a float round trip could only cost it an LSB.
  const Image input = fixture("gradient_8x8_gray.png");
  CHECK(apply_op(input, Op{OpKind::kGrayscale, 0.0F}) == input);
}

TEST_CASE("brightness shifts and saturates", "[cpu]") {
  const Image input = fixture("gradient_8x8_gray.png");  // column x holds 255*x/7

  CHECK(apply_op(input, Op{OpKind::kBrightness, 0.0F}) == input);

  const Image brighter = apply_op(input, Op{OpKind::kBrightness, 0.2F});
  CHECK(std::abs(static_cast<int>(brighter.at(2, 0, 0)) - (72 + 51)) <= 1);
  CHECK(brighter.at(7, 0, 0) == 255);  // 255 + 51 saturates rather than wrapping

  const Image darker = apply_op(input, Op{OpKind::kBrightness, -0.2F});
  CHECK(darker.at(0, 0, 0) == 0);  // 0 - 51 clamps rather than wrapping to 204
  CHECK(std::abs(static_cast<int>(darker.at(5, 0, 0)) - (182 - 51)) <= 1);
}

TEST_CASE("threshold is a hard >= comparison in normalized space", "[cpu]") {
  const Image input = fixture("gradient_8x8_gray.png");
  const Image binary = apply_op(input, Op{OpKind::kThreshold, 0.5F});

  // Column values are 0, 36, 72, 109, 145, 182, 218, 255; the 0.5 cut is at 127.5.
  const std::array<int, 8> expected{0, 0, 0, 0, 255, 255, 255, 255};
  for (int x = 0; x < 8; ++x) {
    CHECK(static_cast<int>(binary.at(x, 3, 0)) == expected[static_cast<std::size_t>(x)]);
  }

  // The boundary is inclusive: a sample exactly at the threshold passes.
  const Image half = image_from(1, 1, 1, {128});
  CHECK(apply_op(half, Op{OpKind::kThreshold, 128.0F / 255.0F}).at(0, 0, 0) == 255);
}

TEST_CASE("gaussian weights are normalized and symmetric", "[cpu]") {
  for (const float sigma : {0.1F, 0.5F, 1.0F, 1.4F, 2.5F, 4.0F}) {
    const std::vector<float> weights = imgjit::cpu::gaussian_weights_1d(sigma);
    const int radius = imgjit::cpu::gaussian_radius(sigma);
    REQUIRE(weights.size() == static_cast<std::size_t>(2 * radius + 1));
    CHECK(std::accumulate(weights.begin(), weights.end(), 0.0F) == Catch::Approx(1.0F));
    for (int i = 1; i <= radius; ++i) {
      CHECK(weights[static_cast<std::size_t>(radius - i)] ==
            Catch::Approx(weights[static_cast<std::size_t>(radius + i)]));
    }
    // The 2D filter is the outer product, so it is normalized iff the 1D one is.
  }
}

TEST_CASE("the gaussian radius stays within the bound the sigma cap buys", "[cpu]") {
  // Phase 3 bakes (2r+1)^2 taps as literals and Phase 7 turns r into a halo, so this
  // is the number the [0.1, 4.0] sigma range in op_chain.h exists to bound.
  CHECK(imgjit::cpu::gaussian_radius(0.1F) == 1);
  CHECK(imgjit::cpu::gaussian_radius(1.0F) == 3);
  CHECK(imgjit::cpu::gaussian_radius(1.4F) == 5);
  CHECK(imgjit::cpu::gaussian_radius(4.0F) == 12);
}

TEST_CASE("a normalized blur leaves a constant image constant", "[cpu]") {
  // Catches both an unnormalized weight table and a mishandled edge: with clamped
  // addressing, border pixels must come out identical to interior ones.
  const Image input = fixture("solid_4x4_rgb.png");
  const Image blurred = apply_op(input, Op{OpKind::kGaussian, 1.4F});
  require_within_one_lsb(blurred, input);
}

TEST_CASE("gaussian matches an independent separable implementation", "[cpu]") {
  for (const std::string& name : {std::string("gradient_8x8_gray.png"),
                                  std::string("checkerboard_16x16_rgb.png"),
                                  std::string("gradient_32x32_rgba.png")}) {
    const Image input = fixture(name);
    for (const float sigma : {0.5F, 1.4F}) {
      require_within_one_lsb(apply_op(input, Op{OpKind::kGaussian, sigma}),
                             separable_gaussian_reference(input, sigma));
    }
  }
}

TEST_CASE("blurring preserves a monotonic gradient", "[cpu]") {
  const Image input = fixture("gradient_8x8_gray.png");
  const Image blurred = apply_op(input, Op{OpKind::kGaussian, 1.4F});
  for (int x = 1; x < blurred.width(); ++x) {
    CHECK(blurred.at(x, 4, 0) >= blurred.at(x - 1, 4, 0));
  }
}

TEST_CASE("sobel is zero on a constant image", "[cpu]") {
  const Image edges = apply_op(fixture("solid_4x4_rgb.png"), Op{OpKind::kSobel, 0.0F});
  for (int y = 0; y < edges.height(); ++y) {
    for (int x = 0; x < edges.width(); ++x) {
      for (int c = 0; c < 3; ++c) {
        CHECK(static_cast<int>(edges.at(x, y, c)) == 0);
      }
    }
  }
}

TEST_CASE("sobel responds to a step edge with the hand-computed magnitude", "[cpu]") {
  // 5x5 gray, left two columns black and the rest white. Rows are identical, so
  // gy is 0 and the magnitude is |gx| alone: 4.0 in normalized units at the two
  // columns straddling the step, saturating to 255, and 0 where the 3x3 window sees
  // one value — including at x=0, where clamped addressing must replicate the border
  // rather than wrap or read a zero.
  std::vector<std::uint8_t> samples(25, 255);
  for (int y = 0; y < 5; ++y) {
    samples[static_cast<std::size_t>(y) * 5 + 0] = 0;
    samples[static_cast<std::size_t>(y) * 5 + 1] = 0;
  }
  const Image edges = apply_op(image_from(5, 5, 1, samples), Op{OpKind::kSobel, 0.0F});

  const std::array<int, 5> expected{0, 255, 255, 0, 0};
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 5; ++x) {
      CHECK(static_cast<int>(edges.at(x, y, 0)) == expected[static_cast<std::size_t>(x)]);
    }
  }
}

TEST_CASE("every op passes alpha through untouched", "[cpu]") {
  const Image input = fixture("gradient_32x32_rgba.png");  // alpha alternates 255/180
  const std::vector<Op> ops{
      {OpKind::kGrayscale, 0.0F}, {OpKind::kInvert, 0.0F},    {OpKind::kBrightness, -0.5F},
      {OpKind::kThreshold, 0.5F}, {OpKind::kGaussian, 1.4F},  {OpKind::kSobel, 0.0F}};

  for (const Op& op : ops) {
    const Image output = apply_op(input, op);
    REQUIRE(output.channels() == 4);
    for (int y = 0; y < input.height(); ++y) {
      for (int x = 0; x < input.width(); ++x) {
        REQUIRE(output.at(x, y, 3) == input.at(x, y, 3));
      }
    }
  }
}

TEST_CASE("every op preserves the frame's dimensions and channel count", "[cpu]") {
  for (const std::string& name :
       {std::string("gradient_8x8_gray.png"), std::string("checkerboard_16x16_rgb.png"),
        std::string("gradient_32x32_rgba.png")}) {
    const Image input = fixture(name);
    for (const imgjit::OpSpec& spec : imgjit::kOpSpecs) {
      const Image output = apply_op(input, Op{spec.kind, spec.default_param});
      CHECK(output.width() == input.width());
      CHECK(output.height() == input.height());
      CHECK(output.channels() == input.channels());
    }
  }
}

TEST_CASE("a chain folds its ops left to right", "[cpu]") {
  const Image input = fixture("checkerboard_16x16_rgb.png");
  const Image stepwise = apply_op(apply_op(apply_op(input, Op{OpKind::kGrayscale, 0.0F}),
                                           Op{OpKind::kGaussian, 1.4F}),
                                  Op{OpKind::kSobel, 0.0F});
  CHECK(chain_of(input, "grayscale,gaussian:1.4,sobel") == stepwise);
}

TEST_CASE("an empty chain is the identity", "[cpu]") {
  const Image input = fixture("checkerboard_16x16_rgb.png");
  CHECK(chain_of(input, "") == input);
}

TEST_CASE("chain order changes the result", "[cpu]") {
  // The reason canonicalization must never reorder: these are different images.
  const Image input = fixture("checkerboard_16x16_rgb.png");
  CHECK_FALSE(chain_of(input, "gaussian:1.4,sobel") == chain_of(input, "sobel,gaussian:1.4"));
}
