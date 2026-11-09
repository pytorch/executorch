# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.26)

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
set(_common_compile_options
    $<$<CXX_COMPILER_ID:MSVC>:/W4
    /WX
    /wd4996>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall
    -Werror
    -Wno-deprecated-declarations
    -fPIC>
)
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

if(EXECUTORCH_BUILD_WASM_WEBGPU)
  if(NOT TARGET webgpu_backend)
    message(
      FATAL_ERROR
        "EXECUTORCH_BUILD_WASM_WEBGPU requires the webgpu_backend target. "
        "Enable EXECUTORCH_BUILD_WEBGPU and add backends/webgpu before "
        "extension/wasm."
    )
  endif()
  list(APPEND link_libraries webgpu_backend)
endif()

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

if(EXECUTORCH_BUILD_WASM_WEBGPU)
  target_compile_definitions(executorch_wasm PUBLIC EXECUTORCH_WASM_WEBGPU)
  # Delegated execution blocks on wgpuInstanceWaitAny, which emdawnwebgpu
  # implements with Asyncify.handleAsync; the port aborts in that call when
  # Asyncify is off. ASYNCIFY_IGNORE_INDIRECT must stay unset: the wait is
  # reached through the virtual backend interface, an indirect call. Model data
  # lives in the Wasm heap before delegate upload, so permit growth beyond
  # Emscripten's 16 MiB default and retain the full wasm32 address-space
  # ceiling.
  target_link_options(
    executorch_wasm PUBLIC -sASYNCIFY -sASYNCIFY_STACK_SIZE=65536
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB
  )
  # The backend reports device and graph failures by throwing, and catches them
  # at its own boundary; Emscripten needs the flag when compiling and when
  # linking.
  target_compile_options(executorch_wasm PUBLIC -fexceptions)
  target_link_options(executorch_wasm PUBLIC -fexceptions)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
