#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_compile_options -Wno-deprecated-declarations)

# -----------------------------------------------------------------------------
# FlatBuffer schema generation
# -----------------------------------------------------------------------------

set(_mlx_schema__include_dir "${CMAKE_BINARY_DIR}/schema/include")
set(_mlx_schema__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/schema.fbs
)

# Paths to headers generated from the .fbs files.
set(_mlx_schema__outputs
    "${_mlx_schema__include_dir}/executorch/backends/apple/mlx/serialization/schema_generated.h"
)

# Generate the headers from the .fbs files.
add_custom_command(
  OUTPUT ${_mlx_schema__outputs}
  COMMAND
    flatc --cpp --cpp-std c++11 --scoped-enums -o
    "${_mlx_schema__include_dir}/executorch/backends/apple/mlx/serialization"
    ${_mlx_schema__srcs}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS flatc ${_mlx_schema__srcs}
  COMMENT "Generating mlx_schema headers"
  VERBATIM
)

add_library(mlx_schema INTERFACE ${_mlx_schema__outputs})
set_target_properties(mlx_schema PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(
  mlx_schema
  INTERFACE
    $<BUILD_INTERFACE:${_mlx_schema__include_dir}>
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/third-party/flatbuffers/include>
)

# -----------------------------------------------------------------------------
# MLX dependency (fetched via FetchContent)
# -----------------------------------------------------------------------------

include(FetchContent)

# MLX build options - we only need the C++ library
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_CPU OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_METAL ON CACHE BOOL "" FORCE)
set(MLX_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_SAFETENSORS OFF CACHE BOOL "" FORCE)
set(MLX_METAL_JIT OFF CACHE BOOL "" FORCE)

# MLX uses FetchContent for json. When FetchContent_MakeAvailable(json) is called,
# it will run add_subdirectory on the json source. ExecuTorch already adds json via
# add_subdirectory(third-party/json) BEFORE this backend is processed.
#
# To prevent the conflict, we patch MLX's CMakeLists.txt to wrap the json fetch
# in a target check. The patch file is in backends/apple/mlx/patches/

# Ensure CMAKE_OSX_DEPLOYMENT_TARGET is set for MLX's version check.
# MLX requires macOS >= 14.0. If not set, default to 14.0.
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version" FORCE)
endif()

FetchContent_Declare(
  mlx
  GIT_REPOSITORY https://github.com/ml-explore/mlx.git
  GIT_TAG v0.30.3
  PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_json.patch || true
)

message(STATUS "Fetching MLX...")
FetchContent_MakeAvailable(mlx)

# -----------------------------------------------------------------------------
# MLX Backend library
# -----------------------------------------------------------------------------

set(_mlx_backend__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/MLXLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/MLXBackend.cpp
)

add_library(mlxdelegate ${_mlx_backend__srcs})

# Ensure schema is generated before compiling
add_dependencies(mlxdelegate mlx_schema flatc)

target_include_directories(
  mlxdelegate
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${_mlx_schema__include_dir}
    ${mlx_SOURCE_DIR}
)

# Link against MLX and executorch
target_link_libraries(
  mlxdelegate
  PRIVATE
    mlx_schema
    executorch_core
    mlx
)

executorch_target_link_options_shared_lib(mlxdelegate)
target_compile_options(mlxdelegate PUBLIC ${_common_compile_options})

install(
  TARGETS mlxdelegate mlx_schema
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

add_subdirectory(test)
