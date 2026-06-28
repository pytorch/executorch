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
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_compile_options -Wall -Werror -Wno-deprecated-declarations)

# Sanitizer flags (asan + ubsan) for security hardening — CI only. Enable via:
# cmake --preset mlx-release -DEXECUTORCH_MLX_ENABLE_SANITIZERS=ON
option(EXECUTORCH_MLX_ENABLE_SANITIZERS
       "Enable ASan + UBSan for MLX delegate and tests" OFF
)
if(EXECUTORCH_MLX_ENABLE_SANITIZERS)
  list(APPEND _common_compile_options -fsanitize=address,undefined
       -fno-omit-frame-pointer
  )
  set(_mlx_sanitizer_link_options -fsanitize=address,undefined)
endif()

# -----------------------------------------------------------------------------
# Code generation from schema.fbs
# -----------------------------------------------------------------------------
#
# The generate.py script generates all code from schema.fbs: Python:
# mlx_graph_schema.py, _generated_serializers.py, _generated/ C++: MLXLoader.h,
# MLXLoader.cpp, schema_generated.h
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
    FATAL_ERROR "MLX submodule not initialized.\n"
                "Run: git submodule update --init backends/mlx/third-party/mlx"
  )
endif()

# Validate deployment target - MLX requires macOS 14.0+ / iOS 17.0+
#
# The macOS preset uses ios.toolchain.cmake (with PLATFORM=MAC_ARM64), so
# DEPLOYMENT_TARGET is set for both macOS and iOS builds. We check PLATFORM to
# distinguish them rather than relying on which variable is set.
set(_mlx_deployment_target_ok TRUE)
if(PLATFORM AND PLATFORM MATCHES "^MAC")
  # macOS build via ios.toolchain.cmake (e.g., MAC_ARM64, MAC_UNIVERSAL)
  if(DEPLOYMENT_TARGET)
    set(_mlx_dt_value ${DEPLOYMENT_TARGET})
  elseif(CMAKE_OSX_DEPLOYMENT_TARGET)
    set(_mlx_dt_value ${CMAKE_OSX_DEPLOYMENT_TARGET})
  endif()
  if(_mlx_dt_value AND _mlx_dt_value VERSION_LESS "14.0")
    set(_mlx_deployment_target_ok FALSE)
    set(_mlx_deployment_target_value ${_mlx_dt_value})
    set(_mlx_deployment_target_min "14.0")
  endif()
elseif(DEPLOYMENT_TARGET)
  # iOS/tvOS/watchOS/visionOS builds via ios.toolchain.cmake
  if(DEPLOYMENT_TARGET VERSION_LESS "17.0")
    set(_mlx_deployment_target_ok FALSE)
    set(_mlx_deployment_target_value ${DEPLOYMENT_TARGET})
    set(_mlx_deployment_target_min "17.0")
  endif()
elseif(CMAKE_OSX_DEPLOYMENT_TARGET)
  # Plain macOS build (no ios.toolchain.cmake)
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

# Build MLX in its own isolated CMake scope via ExternalProject, then consume it
# as a prebuilt static lib through an imported target (see below). Building MLX
# with add_subdirectory would drop its whole project into ExecuTorch's
# target/option namespace, which collides with shared deps MLX fetches (e.g.
# nlohmann_json) and leaks MLX's MLX_BUILD_* options into our cache. The
# isolated scope runs MLX's FetchContent in its own namespace, so no collision
# and no submodule patching are needed.
include(ExternalProject)

set(_mlx_binary_dir ${CMAKE_CURRENT_BINARY_DIR}/mlx)
set(_mlx_static_lib ${_mlx_binary_dir}/libmlx.a)
# With MLX_METAL_JIT=ON, MLX does not install the metallib; it is produced in
# the build tree at this path.
set(_mlx_metallib ${_mlx_binary_dir}/mlx/backend/metal/kernels/mlx.metallib)

message(
  STATUS "Building MLX from submodule (ExternalProject): ${MLX_SOURCE_DIR}"
)
ExternalProject_Add(
  mlx_external
  SOURCE_DIR ${MLX_SOURCE_DIR}
  BINARY_DIR ${_mlx_binary_dir}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
             -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
             -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
             -DPLATFORM=${PLATFORM}
             -DDEPLOYMENT_TARGET=${DEPLOYMENT_TARGET}
             # We only need the static Metal C++ library.
             -DMLX_BUILD_METAL=ON
             -DMLX_BUILD_CPU=OFF
             -DMLX_BUILD_CUDA=OFF
             -DMLX_BUILD_SHARED_LIBS=OFF
             -DMLX_BUILD_PYTHON_BINDINGS=OFF
             -DMLX_BUILD_PYTHON_STUBS=OFF
             -DMLX_BUILD_TESTS=OFF
             -DMLX_BUILD_EXAMPLES=OFF
             -DMLX_BUILD_BENCHMARKS=OFF
             -DMLX_BUILD_GGUF=OFF
             -DMLX_BUILD_SAFETENSORS=OFF
             -DMLX_METAL_JIT=ON
  # MLX's own install() does not emit libmlx.a where we consume it or the
  # metallib at all, so skip the install step and read both from the build tree.
  INSTALL_COMMAND ""
  # Required so Ninja knows these are produced by the external build.
  BUILD_BYPRODUCTS ${_mlx_static_lib} ${_mlx_metallib}
)

# Imported target for the MLX static library produced by mlx_external. A static
# libmlx.a carries no transitive link deps, so re-add the frameworks MLX itself
# links (mirrors third-party/mlx/CMakeLists.txt:209). CPU is OFF, so Accelerate
# is not needed.
add_library(mlx STATIC IMPORTED GLOBAL)
set_target_properties(mlx PROPERTIES IMPORTED_LOCATION ${_mlx_static_lib})
# Headers come from the submodule source tree (always present), not the build
# dir, so mlxdelegate compilation cannot race the external build.
target_include_directories(mlx INTERFACE ${MLX_SOURCE_DIR})
find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(QUARTZ_FRAMEWORK QuartzCore)
set_target_properties(
  mlx
  PROPERTIES INTERFACE_LINK_LIBRARIES
             "${METAL_FRAMEWORK};${FOUNDATION_FRAMEWORK};${QUARTZ_FRAMEWORK}"
)
add_dependencies(mlx mlx_external)

# -----------------------------------------------------------------------------
# MLX Backend library
# -----------------------------------------------------------------------------

# Op logging option (for debugging) - OFF by default for performance
option(ET_MLX_ENABLE_OP_LOGGING "Enable per-op logging in MLX delegate" OFF)

# Custom kernel execution - OFF by default for security. When enabled,
# MetalKernelNode can execute arbitrary Metal shader code embedded in .pte
# files. Only enable for trusted .pte sources.
option(ET_MLX_ALLOW_CUSTOM_KERNEL_EXECUTION
       "Allow MetalKernelNode to execute custom Metal shaders from .pte files"
       ON
)

set(_mlx_backend__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/MLXLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/MLXBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/mlx_mutable_state.cpp
)

add_library(mlxdelegate ${_mlx_backend__srcs})

# Ensure schema is generated before compiling
add_dependencies(mlxdelegate mlx_schema)

# mlx is an imported target, so add_dependencies(mlx mlx_external) above does
# not order the build. Depend on mlx_external directly so libmlx.a exists before
# mlxdelegate links.
add_dependencies(mlxdelegate mlx_external)

# Add logging flag if enabled
if(ET_MLX_ENABLE_OP_LOGGING)
  target_compile_definitions(mlxdelegate PRIVATE ET_MLX_ENABLE_OP_LOGGING=1)
  message(STATUS "MLX delegate op logging ENABLED")
endif()

if(ET_MLX_ALLOW_CUSTOM_KERNEL_EXECUTION)
  target_compile_definitions(
    mlxdelegate PRIVATE ET_MLX_ALLOW_CUSTOM_KERNEL_EXECUTION
  )
  message(STATUS "MLX delegate custom kernel execution ENABLED")
endif()

target_include_directories(
  mlxdelegate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)

# Link against MLX and executorch. mlx is an imported static lib (built by
# mlx_external); it is only consumed at build time here, not re-exported, so
# downstream consumers must link to mlx separately via the package config.
target_link_libraries(
  mlxdelegate PRIVATE mlx_schema executorch_core $<BUILD_INTERFACE:mlx>
)

executorch_target_link_options_shared_lib(mlxdelegate)
target_compile_options(mlxdelegate PRIVATE ${_common_compile_options})
if(EXECUTORCH_MLX_ENABLE_SANITIZERS)
  target_link_options(mlxdelegate PRIVATE ${_mlx_sanitizer_link_options})
endif()

install(
  TARGETS mlxdelegate mlx_schema
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install mlx library for downstream consumers. mlx is an imported target (built
# by mlx_external), so install the static lib file directly rather than via
# install(TARGETS ...), which only accepts built targets.
install(FILES ${_mlx_static_lib} DESTINATION ${CMAKE_INSTALL_LIBDIR})

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
# For the installed library, we put metallib in lib/ alongside libmlx.a. The
# metallib is produced in the mlx_external build tree (MLX_METAL_JIT=ON does not
# install it); _mlx_metallib points there.
install(FILES ${_mlx_metallib} DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Cache the metallib path for pybindings to copy it next to _portable_lib.so
# This enables editable installs to work correctly
set(MLX_METALLIB_PATH
    "${_mlx_metallib}"
    CACHE INTERNAL "Path to mlx.metallib for pybindings"
)

# -----------------------------------------------------------------------------
# Tests (off by default; CI passes -DEXECUTORCH_BUILD_TESTS=ON)
# -----------------------------------------------------------------------------

if(EXECUTORCH_BUILD_TESTS)
  add_subdirectory(test)
endif()
