#include "imgjit/core/image.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace imgjit {

Image::Image(int width, int height, int channels)
    : width_(width),
      height_(height),
      channels_(channels),
      stride_(static_cast<std::size_t>(width) * static_cast<std::size_t>(channels)) {
  if (width <= 0 || height <= 0 || !is_supported_channel_count(channels)) {
    throw std::invalid_argument("Image: bad dimensions " + std::to_string(width) + "x" +
                                std::to_string(height) + "x" + std::to_string(channels));
  }
  pixels_.assign(stride_ * static_cast<std::size_t>(height), 0);
}

Image Image::from_bytes(int width, int height, int channels, const std::uint8_t* source,
                        std::size_t source_stride) {
  Image image(width, height, channels);
  if (source_stride < image.stride_) {
    throw std::invalid_argument("Image::from_bytes: source stride is narrower than a packed row");
  }
  for (int y = 0; y < height; ++y) {
    std::memcpy(image.row(y), source + static_cast<std::size_t>(y) * source_stride, image.stride_);
  }
  return image;
}

}  // namespace imgjit
