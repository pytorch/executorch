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
  COMMAND ${_python_executable} ${_mlx_generate_script} --flatc
          $<TARGET_FILE:flatc>
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS ${_mlx_schema_fbs} ${_mlx_generate_script} flatc
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
# MLX dependency (from submodule)
# -----------------------------------------------------------------------------

set(MLX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party/mlx)

# Check that submodule is initialized
if(NOT EXISTS "${MLX_SOURCE_DIR}/CMakeLists.txt")
  message(
    FATAL_ERROR
      "MLX submodule not initialized.\n"
      "Run: git submodule update --init backends/apple/mlx/third-party/mlx"
  )
endif()

# Validate deployment target - MLX requires macOS 14.0+ / iOS 17.0+ For iOS
# builds using ios.toolchain.cmake, DEPLOYMENT_TARGET is used For macOS builds,
# CMAKE_OSX_DEPLOYMENT_TARGET is used
set(_mlx_deployment_target_ok TRUE)
if(DEPLOYMENT_TARGET)
  # iOS/tvOS/watchOS builds via ios.toolchain.cmake
  if(DEPLOYMENT_TARGET VERSION_LESS "17.0")
    set(_mlx_deployment_target_ok FALSE)
    set(_mlx_deployment_target_value ${DEPLOYMENT_TARGET})
    set(_mlx_deployment_target_min "17.0")
  endif()
elseif(CMAKE_OSX_DEPLOYMENT_TARGET)
  # macOS builds
  if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "14.0")
    set(_mlx_deployment_target_ok FALSE)
    set(_mlx_deployment_target_value ${CMAKE_OSX_DEPLOYMENT_TARGET})
    set(_mlx_deployment_target_min "14.0")
  endif()
endif()

if(NOT _mlx_deployment_target_ok)
  message(
    FATAL_ERROR
      "MLX requires deployment target >= ${_mlx_deployment_target_min}, got ${_mlx_deployment_target_value}.\n"
      "Either increase the deployment target or disable MLX with -DEXECUTORCH_BUILD_MLX=OFF"
  )
endif()

# MLX build options - we only need the C++ library with Metal
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

# Apply JSON patch to prevent conflict with ExecuTorch's nlohmann_json MLX uses
# FetchContent for json; ExecuTorch already has it as submodule
execute_process(
  COMMAND git apply --check ${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_json.patch
  WORKING_DIRECTORY ${MLX_SOURCE_DIR}
  RESULT_VARIABLE _patch_check_result
  OUTPUT_QUIET ERROR_QUIET
)
if(_patch_check_result EQUAL 0)
  execute_process(
    COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_json.patch
    WORKING_DIRECTORY ${MLX_SOURCE_DIR}
    RESULT_VARIABLE _patch_result
  )
  if(_patch_result EQUAL 0)
    message(STATUS "Applied MLX JSON patch")
  endif()
endif()

# Add MLX subdirectory
message(STATUS "Adding MLX from submodule: ${MLX_SOURCE_DIR}")
add_subdirectory(${MLX_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/mlx)

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
                      ${_mlx_schema__include_dir} ${MLX_SOURCE_DIR}
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

# Install mlx library for downstream consumers
install(TARGETS mlx DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Install mlx headers for downstream consumers that may need mlx types
install(
  DIRECTORY ${MLX_SOURCE_DIR}/mlx/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlx
  FILES_MATCHING
  PATTERN "*.h"
)

# Install mlx.metallib (Metal GPU kernels) for runtime execution
#
# MLX searches for metallib in this order (see mlx/backend/metal/device.cpp): 1.
# {binary_dir}/mlx.metallib        - colocated with the .so/.dylib 2.
# {binary_dir}/Resources/mlx/      - Resources subdirectory 3. SwiftPM bundle -
# not applicable for us 4. {binary_dir}/Resources/default/  - Resources
# subdirectory 5. METAL_PATH (compile-time)        - hardcoded build path (won't
# exist)
#
# where {binary_dir} is determined at runtime via dladdr() on the library
# containing MLX code. When MLX is statically linked into _portable_lib.so, this
# is the directory containing _portable_lib.so.
#
# For the installed library, we put metallib in lib/ alongside libmlx.a
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/mlx/mlx/backend/metal/kernels/mlx.metallib
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Cache the metallib path for pybindings to copy it next to _portable_lib.so
# This enables editable installs to work correctly
set(MLX_METALLIB_PATH
    "${CMAKE_CURRENT_BINARY_DIR}/mlx/mlx/backend/metal/kernels/mlx.metallib"
    CACHE INTERNAL "Path to mlx.metallib for pybindings"
)

# -----------------------------------------------------------------------------
# Tests (off by default; CI passes -DEXECUTORCH_BUILD_TESTS=ON)
# -----------------------------------------------------------------------------

if(EXECUTORCH_BUILD_TESTS)
  add_subdirectory(test)
endif()
