# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example CMakeLists.txt for building executor_runner with Developer Tools
# support. In this example we link devtools and bundled_program libraries into
# executor_runner binary
cmake_minimum_required(VERSION 3.19)
project(devtools_example)

option(EXECUTORCH_BUILD_COREML "Build the Core ML backend" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Find prebuilt libraries. executorch package should contain portable_ops_lib,
# etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
target_link_options_shared_lib(executorch)
target_link_options_shared_lib(portable_ops_lib)

target_include_directories(executorch INTERFACE ${_common_include_directories})

find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../third-party
)

add_executable(example_runner example_runner/example_runner.cpp)
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)

target_include_directories(
  etdump INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/../../devtools/include
                   ${EXECUTORCH_ROOT}/third-party/flatcc/include
)
target_link_libraries(
  example_runner
  executorch
  gflags
  etdump
  extension_data_loader
  bundled_program
  flatccrt
  portable_ops_lib
  portable_kernels
)

if(EXECUTORCH_BUILD_COREML)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  find_library(COREML_FRAMEWORK CoreML)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  find_library(SQLITE_LIBRARY sqlite3)

  set(PROTOBUF_LIB_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/../../backends/apple/coreml/third-party/coremltools/deps/protobuf/cmake
  )
  find_library(
    PROTOBUF_LITE REQUIRED
    NAMES libprotobuf-lite.a
    PATHS ${PROTOBUF_LIB_DIR}
    NO_DEFAULT_PATH
  )

  target_link_libraries(
    example_runner "-Wl,-force_load" coremldelegate
  )

  target_link_libraries(
    example_runner ${PROTOBUF_LITE} ${ACCELERATE_FRAMEWORK}
    ${COREML_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${SQLITE_LIBRARY}
  )
endif()
