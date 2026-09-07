#pragma once

// PNG load/save for the CLI and the tests. File-side convenience only: the server
// never decodes an image from network input, where the payload is always raw
// contiguous pixels (docs/PROTOCOL.md, "Explicit non-goals"). The encode half is
// reachable from the server's flags-bit-1 write path, which is file output, not wire
// format.
//
// Both throw std::runtime_error on failure — these are CLI/test paths, where an
// exception carrying stb's reason is more useful than a status code.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <string>

#include "imgjit/core/image.h"

namespace imgjit {

// Loads with stb's native channel count (1, 3 or 4); 2-channel gray+alpha files are
// expanded to RGBA, since the protocol has no 2-channel format.
Image load_png(const std::string& path);

void save_png(const std::string& path, const Image& image);

}  // namespace imgjit
