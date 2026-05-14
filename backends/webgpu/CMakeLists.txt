# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Ensure vulkan_schema is available even when EXECUTORCH_BUILD_VULKAN is OFF.
# The WebGPU backend reuses the Vulkan FlatBuffer serialization format.
if(NOT TARGET vulkan_schema)
  # We need the schema generation from the Vulkan backend. Build only the schema
  # target by including the Vulkan CMakeLists.txt. The full Vulkan backend will
  # only build if EXECUTORCH_BUILD_VULKAN is ON (which gates the vulkan_backend
  # target), but vulkan_schema is unconditionally defined.
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan
    ${CMAKE_CURRENT_BINARY_DIR}/_vulkan_schema
  )
endif()

set(WEBGPU_SRCS
    runtime/WebGPUBackend.cpp runtime/WebGPUGraph.cpp
    runtime/WebGPUDelegateHeader.cpp runtime/WebGPUDevice.cpp
    runtime/ops/OperatorRegistry.cpp runtime/ops/add/BinaryOp.cpp
)

add_library(webgpu_backend ${WEBGPU_SRCS})

target_include_directories(
  webgpu_backend PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)

target_link_libraries(webgpu_backend PRIVATE vulkan_schema executorch_core)

# Native build: link against wgpu-native
set(WGPU_NATIVE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/wgpu-native"
    CACHE PATH "Path to wgpu-native installation"
)

if(NOT EXISTS "${WGPU_NATIVE_DIR}/lib/libwgpu_native.a")
  message(FATAL_ERROR "wgpu-native not found at ${WGPU_NATIVE_DIR}. "
                      "Run: bash backends/webgpu/scripts/setup-wgpu-native.sh"
  )
endif()

add_library(wgpu_native STATIC IMPORTED)
set_target_properties(
  wgpu_native PROPERTIES IMPORTED_LOCATION
                         "${WGPU_NATIVE_DIR}/lib/libwgpu_native.a"
)

target_include_directories(
  webgpu_backend PUBLIC $<BUILD_INTERFACE:${WGPU_NATIVE_DIR}/include>
)
target_link_libraries(webgpu_backend PRIVATE wgpu_native)

if(APPLE)
  target_link_libraries(
    webgpu_backend PRIVATE "-framework Metal" "-framework QuartzCore"
                           "-framework CoreGraphics" "-framework Foundation"
  )
else()
  target_link_libraries(webgpu_backend PRIVATE dl m pthread)
endif()

target_compile_options(webgpu_backend PRIVATE -fexceptions)

# Link with --whole-archive for static registration of backend + ops
executorch_target_link_options_shared_lib(webgpu_backend)

set_property(TARGET webgpu_backend PROPERTY CXX_STANDARD 17)

install(
  TARGETS webgpu_backend
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Native test target
if(EXECUTORCH_BUILD_WEBGPU_TEST)
  add_executable(webgpu_native_test test/test_webgpu_native.cpp)

  target_include_directories(
    webgpu_native_test PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
                               "${WGPU_NATIVE_DIR}/include"
  )

  target_link_libraries(
    webgpu_native_test
    PRIVATE webgpu_backend
            wgpu_native
            executorch_core
            extension_module_static
            extension_data_loader
            extension_tensor
            portable_kernels
            portable_ops_lib
  )

  if(APPLE)
    target_link_libraries(
      webgpu_native_test PRIVATE "-framework Metal" "-framework QuartzCore"
                                 "-framework CoreGraphics"
    )
  else()
    target_link_libraries(webgpu_native_test PRIVATE dl m pthread)
  endif()

  target_compile_options(webgpu_native_test PRIVATE -fexceptions)
  set_property(TARGET webgpu_native_test PROPERTY CXX_STANDARD 17)
endif()
