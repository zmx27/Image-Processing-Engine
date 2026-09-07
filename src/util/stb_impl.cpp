// Single translation unit that instantiates the vendored header-only stb libraries,
// so no other file has to define the implementation macros. Portable: no CUDA, and
// it builds on macOS with no toolkit present.
//
// stb is a file-side convenience only — it never decodes network input
// (docs/PROTOCOL.md, "Explicit non-goals").

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
