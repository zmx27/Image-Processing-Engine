#include "imgjit/util/image_io.h"

#include <stdexcept>

#include "stb_image.h"
#include "stb_image_write.h"

namespace imgjit {

Image load_png(const std::string& path) {
  int width = 0;
  int height = 0;
  int file_channels = 0;
  // Pass 0 to keep the file's own channel count, except for gray+alpha: the protocol
  // knows 1, 3 and 4 only, so ask stb to expand those two channels to RGBA.
  if (stbi_info(path.c_str(), &width, &height, &file_channels) == 0) {
    throw std::runtime_error("load_png: cannot read " + path + ": " + stbi_failure_reason());
  }
  const int requested = file_channels == 2 ? 4 : 0;

  int loaded_channels = 0;
  std::uint8_t* pixels = stbi_load(path.c_str(), &width, &height, &loaded_channels, requested);
  if (pixels == nullptr) {
    throw std::runtime_error("load_png: cannot read " + path + ": " + stbi_failure_reason());
  }
  const int channels = requested != 0 ? requested : loaded_channels;

  try {
    Image image = Image::from_bytes(width, height, channels, pixels,
                                    static_cast<std::size_t>(width) *
                                        static_cast<std::size_t>(channels));
    stbi_image_free(pixels);
    return image;
  } catch (...) {
    stbi_image_free(pixels);
    throw;
  }
}

void save_png(const std::string& path, const Image& image) {
  if (image.empty()) {
    throw std::runtime_error("save_png: refusing to write an empty image to " + path);
  }
  if (stbi_write_png(path.c_str(), image.width(), image.height(), image.channels(), image.data(),
                     static_cast<int>(image.stride())) == 0) {
    throw std::runtime_error("save_png: cannot write " + path);
  }
}

}  // namespace imgjit
