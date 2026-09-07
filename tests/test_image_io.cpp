// PNG load/save (docs/PLAN.md Phase 2). File-side only — nothing here is ever on the
// wire (docs/PROTOCOL.md, "Explicit non-goals").

#include <string>

#include "catch_amalgamated.hpp"
#include "imgjit/util/image_io.h"

using imgjit::Image;

namespace {

std::string fixture_path(const std::string& name) {
  return std::string(IMGJIT_TESTDATA_DIR) + "/" + name;
}

std::string temp_path(const std::string& name) {
  return std::string(IMGJIT_TEST_TMP_DIR) + "/" + name;
}

}  // namespace

TEST_CASE("fixtures load with their own dimensions and channel count", "[io]") {
  struct Expected {
    const char* name;
    int width;
    int height;
    int channels;
  };
  const Expected fixtures[] = {
      {"solid_4x4_rgb.png", 4, 4, 3},
      {"gradient_8x8_gray.png", 8, 8, 1},
      {"checkerboard_16x16_rgb.png", 16, 16, 3},
      {"gradient_32x32_rgba.png", 32, 32, 4},
  };

  for (const Expected& expected : fixtures) {
    const Image image = imgjit::load_png(fixture_path(expected.name));
    CHECK(image.width() == expected.width);
    CHECK(image.height() == expected.height);
    CHECK(image.channels() == expected.channels);
    CHECK(image.stride() ==
          static_cast<std::size_t>(expected.width) * static_cast<std::size_t>(expected.channels));
    CHECK(image.byte_count() == image.stride() * static_cast<std::size_t>(expected.height));
  }
}

TEST_CASE("save then load round-trips every pixel", "[io]") {
  // PNG is lossless, so this is exact — if it ever is not, the stride passed to
  // stb_image_write is wrong.
  for (const char* name : {"gradient_8x8_gray.png", "checkerboard_16x16_rgb.png",
                           "gradient_32x32_rgba.png"}) {
    const Image original = imgjit::load_png(fixture_path(name));
    const std::string written = temp_path(std::string("roundtrip_") + name);
    imgjit::save_png(written, original);
    CHECK(imgjit::load_png(written) == original);
  }
}

TEST_CASE("a missing or unwritable file is an error, not a crash", "[io]") {
  CHECK_THROWS(imgjit::load_png(fixture_path("no_such_fixture.png")));
  CHECK_THROWS(imgjit::save_png(temp_path("empty.png"), Image{}));
  CHECK_THROWS(imgjit::save_png(temp_path("no_such_directory/out.png"),
                                imgjit::load_png(fixture_path("solid_4x4_rgb.png"))));
}

TEST_CASE("fixture content is what the ops tests assume", "[io]") {
  // These exact values are hand-computed into test_cpu_ops.cpp's expectations, so a
  // regenerated fixture must fail here rather than as a puzzling filter failure.
  const Image solid = imgjit::load_png(fixture_path("solid_4x4_rgb.png"));
  CHECK(static_cast<int>(solid.at(2, 3, 0)) == 200);
  CHECK(static_cast<int>(solid.at(2, 3, 1)) == 40);
  CHECK(static_cast<int>(solid.at(2, 3, 2)) == 40);

  const Image gradient = imgjit::load_png(fixture_path("gradient_8x8_gray.png"));
  for (int x = 0; x < 8; ++x) {
    CHECK(static_cast<int>(gradient.at(x, 0, 0)) == 255 * x / 7);
  }
}
