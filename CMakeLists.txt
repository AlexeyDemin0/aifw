
# ── In-source build guard ─────────────────────────────────────────────────────

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed.\n"
    "Please create a separate build directory:\n"
    "  cmake -B build"
  )
endif()

# ──────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.20)

project(aifw
  VERSION 0.1.0
  DESCRIPTION "AI framework"
  LANGUAGES CXX
)

# ── Standards ─────────────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build type ───────────────────────────────────────────────────────────────

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ─── Warnings interface ───────────────────────────────────────────────────────

add_library(aifw_warnings INTERFACE)
add_library(aifw::warnings ALIAS aifw_warnings)

target_compile_options(aifw_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
  >
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4
  >
)

# ─── Options ──────────────────────────────────────────────────────────────────

option(AIFW_BUILD_TESTS "Build tests" ON)
option(AIFW_ENABLE_CUDA "Enable CUDA backend" OFF)

# ─── CUDA ─────────────────────────────────────────────────────────────────────

if(AIFW_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolKit REQUIRED)
  message(STATUS "CUDA enabled: ${CUDAToolKit_VERSION}")
endif()

# ─── Subdirectories ───────────────────────────────────────────────────────────

add_subdirectory(aifw)

if(AIFW_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
