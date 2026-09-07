# The one translation unit that instantiates stb. Warnings are disabled for this
# source only: it is vendored third-party code and the project builds -Werror.
add_library(imgjit_stb STATIC stb_impl.cpp)
target_include_directories(imgjit_stb PUBLIC "${CMAKE_SOURCE_DIR}/third_party/stb")
set_source_files_properties(stb_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")

# PNG load/save for the CLI and tests. Separate from imgjit_core so that the server
# and the backends never link an image codec at all — nothing decodes network input
# (docs/PROTOCOL.md, "Explicit non-goals").
add_library(imgjit_io STATIC image_io.cpp)
target_link_libraries(imgjit_io PUBLIC imgjit_core PRIVATE imgjit_stb)
