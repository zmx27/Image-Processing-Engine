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

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ASan and TSan are TWO BUILD CONFIGURATIONS, NOT ONE (docs/PLAN.md Phase 4): they
# cannot be linked into the same binary, so the Phase 4 gate is two build trees over the
# same test set rather than one instrumented build.
#
#   cmake -B build-asan -DIMGJIT_SANITIZER=address && cmake --build build-asan -j
#   ctest --test-dir build-asan --output-on-failure
#   cmake -B build-tsan -DIMGJIT_SANITIZER=thread  && cmake --build build-tsan -j
#   ctest --test-dir build-tsan --output-on-failure
set(IMGJIT_SANITIZER "off" CACHE STRING "Sanitizer to build with: off, address, or thread")
set_property(CACHE IMGJIT_SANITIZER PROPERTY STRINGS off address thread)
if(IMGJIT_SANITIZER STREQUAL "address")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
  message(STATUS "imgjit: AddressSanitizer build")
elseif(IMGJIT_SANITIZER STREQUAL "thread")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=thread)
  message(STATUS "imgjit: ThreadSanitizer build")
elseif(NOT IMGJIT_SANITIZER STREQUAL "off")
  message(FATAL_ERROR "IMGJIT_SANITIZER must be one of: off, address, thread")
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
