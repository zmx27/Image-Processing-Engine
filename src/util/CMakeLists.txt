# The one translation unit that instantiates stb. Warnings are disabled for this
# source only: it is vendored third-party code and the project builds -Werror.
add_library(imgjit_stb STATIC stb_impl.cpp)
target_include_directories(imgjit_stb PUBLIC "${CMAKE_SOURCE_DIR}/third_party/stb")
set_source_files_properties(stb_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")
