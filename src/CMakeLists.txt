# Portable subdirectories build everywhere, including macOS with no CUDA toolkit
# present; src/backend/cuda is gated behind IMGJIT_ENABLE_CUDA. See CLAUDE.md.
add_subdirectory(util)

if(IMGJIT_ENABLE_CUDA)
  add_subdirectory(backend/cuda)
endif()
