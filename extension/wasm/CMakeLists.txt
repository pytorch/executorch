# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.29)

project(executorch_wasm)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Emscripten is required to build this target")
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
set(_common_compile_options -Wno-deprecated-declarations -fPIC -Wall -Werror)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(link_libraries)
list(
  APPEND
  link_libraries
  embind
  executorch_core
  extension_data_loader
  extension_module_static
  extension_tensor
  extension_runner_util
)

add_library(executorch_wasm OBJECT wasm_bindings.cpp)

if(EXECUTORCH_ENABLE_EVENT_TRACER)
  list(APPEND link_libraries etdump)
  target_compile_definitions(
    executorch_wasm PUBLIC EXECUTORCH_ENABLE_EVENT_TRACER
  )
endif()

target_compile_options(executorch_wasm PUBLIC ${_common_compile_options})
target_include_directories(
  executorch_wasm PUBLIC ${_common_include_directories}
)
target_link_libraries(
  executorch_wasm
  PUBLIC ${link_libraries}
  INTERFACE executorch_kernels
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
