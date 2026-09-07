cmake_minimum_required(VERSION 3.24)
project(imgjit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Werror)
endif()

# No Mac has an NVIDIA GPU (Intel or Apple Silicon) — CUDA is only ever found on
# Colab/Linux. This must stay optional so the portable core builds on macOS with
# no CUDA toolkit present at all. See CLAUDE.md and docs/ARCHITECTURE.md.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(IMGJIT_ENABLE_CUDA ON)
  message(STATUS "imgjit: CUDA toolkit found (${CUDAToolkit_VERSION}) — CUDA backend enabled")
else()
  set(IMGJIT_ENABLE_CUDA OFF)
  message(STATUS "imgjit: no CUDA toolkit found — building portable core only (CPU backend)")
endif()

enable_testing()

# Populated phase by phase per docs/PLAN.md: src/core, src/net, src/util, and
# src/backend/cpu are portable and build with no CUDA toolkit present;
# src/backend/cuda is gated behind IMGJIT_ENABLE_CUDA. Guarded with EXISTS so
# this file configures cleanly before those directories are populated.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/CMakeLists.txt")
  add_subdirectory(src)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt")
  add_subdirectory(bench)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/CMakeLists.txt")
  add_subdirectory(tools)
endif()
