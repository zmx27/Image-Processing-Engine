# The one translation unit that instantiates stb. Warnings are disabled for this
# source only: it is vendored third-party code and the project builds -Werror.
add_library(imgjit_stb STATIC stb_impl.cpp)
target_include_directories(imgjit_stb PUBLIC "${CMAKE_SOURCE_DIR}/third_party/stb")
set_source_files_properties(stb_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")

# PNG load/save for the CLI and tests. Separate from imgjit_core so that the backends
# never link an image codec at all, and so the server links only the ENCODE half it
# needs for the flags-bit-1 write path — nothing anywhere decodes network input
# (docs/PROTOCOL.md, "Explicit non-goals").
add_library(imgjit_io STATIC image_io.cpp)
target_link_libraries(imgjit_io PUBLIC imgjit_core PRIVATE imgjit_stb)

# Concurrency plumbing shared by the server: the frame slot pool, and the bounded queue
# (header-only, hence no source of its own). Separate from imgjit_net because neither
# knows anything about sockets, and because Phase 5 hands the pool pinned memory from
# the CUDA backend without either of them changing.
add_library(imgjit_util STATIC slot_pool.cpp)
target_link_libraries(imgjit_util PUBLIC imgjit_core Threads::Threads)
