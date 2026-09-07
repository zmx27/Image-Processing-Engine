#include "imgjit/backend/cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace imgjit::cpu {
namespace {

float load_sample(std::uint8_t value) { return static_cast<float>(value) / 255.0F; }

std::uint8_t store_sample(float value) {
  return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
}

int clamp_coordinate(int value, int limit) { return std::clamp(value, 0, limit - 1); }

// Copies alpha through unchanged. Every op calls this instead of open-coding the
// channel==4 special case, so "alpha is never touched" holds in exactly one place.
void copy_alpha(const Image& input, Image& output) {
  if (input.channels() != 4) {
    return;
  }
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      output.at(x, y, 3) = input.at(x, y, 3);
    }
  }
}

Image apply_grayscale(const Image& input) {
  Image output(input.width(), input.height(), input.channels());
  const int colors = color_channels(input.channels());
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      // A 1-channel image is already luminance; leave it bit-identical rather than
      // pushing it through a float round trip that could shift it by an LSB.
      if (colors == 1) {
        output.at(x, y, 0) = input.at(x, y, 0);
        continue;
      }
      const float luma = kLumaRed * load_sample(input.at(x, y, 0)) +
                         kLumaGreen * load_sample(input.at(x, y, 1)) +
                         kLumaBlue * load_sample(input.at(x, y, 2));
      const std::uint8_t gray = store_sample(luma);
      for (int c = 0; c < colors; ++c) {
        output.at(x, y, c) = gray;
      }
    }
  }
  copy_alpha(input, output);
  return output;
}

// Integer op, deliberately: no float round trip, so GPU and CPU compare exactly.
Image apply_invert(const Image& input) {
  Image output(input.width(), input.height(), input.channels());
  const int colors = color_channels(input.channels());
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      for (int c = 0; c < colors; ++c) {
        output.at(x, y, c) = static_cast<std::uint8_t>(255 - input.at(x, y, c));
      }
    }
  }
  copy_alpha(input, output);
  return output;
}

Image apply_brightness(const Image& input, float delta) {
  Image output(input.width(), input.height(), input.channels());
  const int colors = color_channels(input.channels());
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      for (int c = 0; c < colors; ++c) {
        output.at(x, y, c) = store_sample(load_sample(input.at(x, y, c)) + delta);
      }
    }
  }
  copy_alpha(input, output);
  return output;
}

Image apply_threshold(const Image& input, float threshold) {
  Image output(input.width(), input.height(), input.channels());
  const int colors = color_channels(input.channels());
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      for (int c = 0; c < colors; ++c) {
        output.at(x, y, c) = load_sample(input.at(x, y, c)) >= threshold ? 255 : 0;
      }
    }
  }
  copy_alpha(input, output);
  return output;
}

// Direct 2D convolution with the outer product of the 1D weights — not a separable
// two-pass blur. The generated CUDA kernel is a single 2D stencil, and running the
// oracle as two passes would put a float-reassociation difference between them for no
// reason. Both factorizations sum to one, so this is the same filter, evaluated the
// same way the GPU will evaluate it.
Image apply_gaussian(const Image& input, float sigma) {
  const std::vector<float> weights = gaussian_weights_1d(sigma);
  const int radius = gaussian_radius(sigma);

  Image output(input.width(), input.height(), input.channels());
  const int colors = color_channels(input.channels());
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      for (int c = 0; c < colors; ++c) {
        float sum = 0.0F;
        for (int dy = -radius; dy <= radius; ++dy) {
          const int sy = clamp_coordinate(y + dy, input.height());
          const float wy = weights[static_cast<std::size_t>(dy + radius)];
          for (int dx = -radius; dx <= radius; ++dx) {
            const int sx = clamp_coordinate(x + dx, input.width());
            const float wx = weights[static_cast<std::size_t>(dx + radius)];
            sum += wy * wx * load_sample(input.at(sx, sy, c));
          }
        }
        output.at(x, y, c) = store_sample(sum);
      }
    }
  }
  copy_alpha(input, output);
  return output;
}

Image apply_sobel(const Image& input) {
  constexpr float kGx[3][3] = {{-1.0F, 0.0F, 1.0F}, {-2.0F, 0.0F, 2.0F}, {-1.0F, 0.0F, 1.0F}};
  constexpr float kGy[3][3] = {{-1.0F, -2.0F, -1.0F}, {0.0F, 0.0F, 0.0F}, {1.0F, 2.0F, 1.0F}};

  Image output(input.width(), input.height(), input.channels());
  const int colors = color_channels(input.channels());
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      for (int c = 0; c < colors; ++c) {
        float gx = 0.0F;
        float gy = 0.0F;
        for (int dy = -1; dy <= 1; ++dy) {
          const int sy = clamp_coordinate(y + dy, input.height());
          for (int dx = -1; dx <= 1; ++dx) {
            const int sx = clamp_coordinate(x + dx, input.width());
            const float sample = load_sample(input.at(sx, sy, c));
            gx += kGx[dy + 1][dx + 1] * sample;
            gy += kGy[dy + 1][dx + 1] * sample;
          }
        }
        output.at(x, y, c) = store_sample(std::sqrt(gx * gx + gy * gy));
      }
    }
  }
  copy_alpha(input, output);
  return output;
}

}  // namespace

int gaussian_radius(float sigma) {
  return std::max(1, static_cast<int>(std::ceil(3.0F * sigma)));
}

std::vector<float> gaussian_weights_1d(float sigma) {
  const int radius = gaussian_radius(sigma);
  std::vector<float> weights(static_cast<std::size_t>(2 * radius + 1));
  float total = 0.0F;
  for (int i = -radius; i <= radius; ++i) {
    const float value =
        std::exp(-static_cast<float>(i * i) / (2.0F * sigma * sigma));
    weights[static_cast<std::size_t>(i + radius)] = value;
    total += value;
  }
  for (float& weight : weights) {
    weight /= total;
  }
  return weights;
}

Image apply_op(const Image& input, const Op& op) {
  switch (op.kind) {
    case OpKind::kGrayscale:
      return apply_grayscale(input);
    case OpKind::kInvert:
      return apply_invert(input);
    case OpKind::kBrightness:
      return apply_brightness(input, op.param);
    case OpKind::kThreshold:
      return apply_threshold(input, op.param);
    case OpKind::kGaussian:
      return apply_gaussian(input, op.param);
    case OpKind::kSobel:
      return apply_sobel(input);
  }
  throw std::invalid_argument("apply_op: unknown OpKind");
}

Image apply_chain(const Image& input, const OpChain& chain) {
  Image current = input;
  for (const Op& op : chain.ops) {
    current = apply_op(current, op);
  }
  return current;
}

}  // namespace imgjit::cpu
