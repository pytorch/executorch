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
# Code generation from schema.fbs
# -----------------------------------------------------------------------------
#
# The generate.py script generates all code from schema.fbs: - Python:
# mlx_graph_schema.py, _generated_serializers.py, _generated/ - C++:
# MLXLoader.h, MLXLoader.cpp, schema_generated.h
#
# We run generate.py at build time so these files don't need to be checked in.
# -----------------------------------------------------------------------------

set(_mlx_generate_script
    "${CMAKE_CURRENT_SOURCE_DIR}/serialization/generate.py"
)
set(_mlx_schema_fbs "${CMAKE_CURRENT_SOURCE_DIR}/serialization/schema.fbs")

# Generated C++ files that we need for compilation
set(_mlx_generated_cpp_files
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/schema_generated.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/MLXLoader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/MLXLoader.cpp"
)

# Generated Python files (tracked for dependency purposes)
set(_mlx_generated_python_files
    "${CMAKE_CURRENT_SOURCE_DIR}/serialization/mlx_graph_schema.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/serialization/_generated_serializers.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/serialization/_generated_parsers.py"
)

# Run generate.py to create all generated files from schema.fbs Find Python -
# prefer Python3_EXECUTABLE if set (from FindPython3), otherwise use
# PYTHON_EXECUTABLE
if(Python3_EXECUTABLE)
  set(_python_executable ${Python3_EXECUTABLE})
elseif(PYTHON_EXECUTABLE)
  set(_python_executable ${PYTHON_EXECUTABLE})
else()
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )
  set(_python_executable ${Python3_EXECUTABLE})
endif()

add_custom_command(
  OUTPUT ${_mlx_generated_cpp_files} ${_mlx_generated_python_files}
  COMMAND ${_python_executable} ${_mlx_generate_script}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS ${_mlx_schema_fbs} ${_mlx_generate_script}
  COMMENT "Generating MLX delegate code from schema.fbs"
  VERBATIM
)

# Custom target to trigger generation
add_custom_target(
  mlx_generate_code DEPENDS ${_mlx_generated_cpp_files}
                            ${_mlx_generated_python_files}
)

# Interface library for schema includes
add_library(mlx_schema INTERFACE)
add_dependencies(mlx_schema mlx_generate_code)
target_include_directories(
  mlx_schema
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime>
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/third-party/flatbuffers/include>
)

# -----------------------------------------------------------------------------
# MLX dependency (fetched via FetchContent)
# -----------------------------------------------------------------------------

include(FetchContent)

# MLX build options - we only need the C++ library
set(MLX_BUILD_PYTHON_BINDINGS
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_EXAMPLES
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_BENCHMARKS
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_PYTHON_STUBS
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_CUDA
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_CPU
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_METAL
    ON
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_GGUF
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_BUILD_SAFETENSORS
    OFF
    CACHE BOOL "" FORCE
)
set(MLX_METAL_JIT
    ON
    CACHE BOOL "Use JIT compiled Metal kernels"
)

# MLX uses FetchContent for json. When FetchContent_MakeAvailable(json) is
# called, it will run add_subdirectory on the json source. ExecuTorch already
# adds json via add_subdirectory(third-party/json) BEFORE this backend is
# processed.
#
# To prevent the conflict, we patch MLX's CMakeLists.txt to wrap the json fetch
# in a target check. The patch file is in backends/apple/mlx/patches/

# Ensure CMAKE_OSX_DEPLOYMENT_TARGET is set for MLX's version check. MLX
# requires macOS >= 14.0. If not set, default to 14.0.
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "14.0"
      CACHE STRING "Minimum macOS version" FORCE
  )
endif()

FetchContent_Declare(
  mlx
  GIT_REPOSITORY https://github.com/ml-explore/mlx.git
  GIT_TAG v0.30.3
  PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_json.patch ||
                true
)

message(STATUS "Fetching MLX...")
FetchContent_MakeAvailable(mlx)

# -----------------------------------------------------------------------------
# MLX Backend library
# -----------------------------------------------------------------------------

# Op logging option (for debugging) - OFF by default for performance
option(ET_MLX_ENABLE_OP_LOGGING "Enable per-op logging in MLX delegate" OFF)

set(_mlx_backend__srcs ${CMAKE_CURRENT_SOURCE_DIR}/runtime/MLXLoader.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/runtime/MLXBackend.cpp
)

add_library(mlxdelegate ${_mlx_backend__srcs})

# Ensure schema is generated before compiling
add_dependencies(mlxdelegate mlx_schema flatc)

# Add logging flag if enabled
if(ET_MLX_ENABLE_OP_LOGGING)
  target_compile_definitions(mlxdelegate PRIVATE ET_MLX_ENABLE_OP_LOGGING=1)
  message(STATUS "MLX delegate op logging ENABLED")
endif()

target_include_directories(
  mlxdelegate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime
                      ${_mlx_schema__include_dir} ${mlx_SOURCE_DIR}
)

# Link against MLX and executorch mlx is only available at BUILD_INTERFACE -
# consumers must link to mlx separately
target_link_libraries(
  mlxdelegate PRIVATE mlx_schema executorch_core $<BUILD_INTERFACE:mlx>
)

executorch_target_link_options_shared_lib(mlxdelegate)
target_compile_options(mlxdelegate PUBLIC ${_common_compile_options})

install(
  TARGETS mlxdelegate mlx_schema
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install mlx library (from FetchContent) for downstream consumers Since mlx is
# a FetchContent dependency, we need to explicitly install it so that consumers
# of mlxdelegate can link against it
install(TARGETS mlx DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Install mlx headers for downstream consumers that may need mlx types
install(
  DIRECTORY ${mlx_SOURCE_DIR}/mlx/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlx
  FILES_MATCHING
  PATTERN "*.h"
)

# Install mlx.metallib (Metal GPU kernels) for runtime execution MLX searches
# for this in: Resources/mlx/, Resources/default/, or a hardcoded build path We
# install it to lib/ alongside libmlx.a and create a Resources symlink structure
install(FILES ${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

add_subdirectory(test)
