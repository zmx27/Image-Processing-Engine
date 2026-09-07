#pragma once

// The one image representation the whole project uses: 8-bit samples, 1/3/4
// interleaved channels, row-major, with an explicit row stride.
//
// Stride is a first-class field rather than an implied `width * channels` so that
// indexing already goes through it when a padded source appears — the wire payload
// is packed (docs/PROTOCOL.md) but stb and future pitched device copies are not.
// Construction always allocates packed rows; `from_bytes` is what accepts a padded
// source.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imgjit {

class Image {
 public:
  Image() = default;

  // Allocates `height` packed rows, zero-filled.
  Image(int width, int height, int channels);

  // Copies `height` rows of `width * channels` bytes out of `source`, whose rows are
  // `source_stride` bytes apart, into a packed image.
  static Image from_bytes(int width, int height, int channels, const std::uint8_t* source,
                          std::size_t source_stride);

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  std::size_t stride() const { return stride_; }
  std::size_t byte_count() const { return pixels_.size(); }
  bool empty() const { return pixels_.empty(); }

  std::uint8_t* data() { return pixels_.data(); }
  const std::uint8_t* data() const { return pixels_.data(); }

  std::uint8_t* row(int y) { return pixels_.data() + static_cast<std::size_t>(y) * stride_; }
  const std::uint8_t* row(int y) const {
    return pixels_.data() + static_cast<std::size_t>(y) * stride_;
  }

  // Sample `channel` of the pixel at (x, y). No bounds checking — callers in the op
  // loops clamp coordinates themselves (see clamped edge addressing in cpu/ops.h).
  std::uint8_t& at(int x, int y, int channel) {
    return row(y)[static_cast<std::size_t>(x) * static_cast<std::size_t>(channels_) +
                  static_cast<std::size_t>(channel)];
  }
  const std::uint8_t& at(int x, int y, int channel) const {
    return row(y)[static_cast<std::size_t>(x) * static_cast<std::size_t>(channels_) +
                  static_cast<std::size_t>(channel)];
  }

  friend bool operator==(const Image& lhs, const Image& rhs) {
    return lhs.width_ == rhs.width_ && lhs.height_ == rhs.height_ &&
           lhs.channels_ == rhs.channels_ && lhs.stride_ == rhs.stride_ &&
           lhs.pixels_ == rhs.pixels_;
  }

 private:
  int width_{0};
  int height_{0};
  int channels_{0};
  std::size_t stride_{0};
  std::vector<std::uint8_t> pixels_;
};

// 1 (gray), 3 (RGB) and 4 (RGBA) are the only channel counts on the wire
// (docs/PROTOCOL.md) and therefore the only ones any op has to handle.
constexpr bool is_supported_channel_count(int channels) {
  return channels == 1 || channels == 3 || channels == 4;
}

}  // namespace imgjit
