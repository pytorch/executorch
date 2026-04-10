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
  # We need the schema generation from the Vulkan backend. Build only the
  # schema target by including the Vulkan CMakeLists.txt. The full Vulkan
  # backend will only build if EXECUTORCH_BUILD_VULKAN is ON (which gates the
  # vulkan_backend target), but vulkan_schema is unconditionally defined.
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan
    ${CMAKE_CURRENT_BINARY_DIR}/_vulkan_schema
  )
endif()

set(WEBGPU_SRCS
    runtime/WebGPUBackend.cpp
    runtime/WebGPUGraph.cpp
    runtime/WebGPUDelegateHeader.cpp
    runtime/ops/OperatorRegistry.cpp
    runtime/ops/add/BinaryOp.cpp
)

add_library(webgpu_backend ${WEBGPU_SRCS})

target_include_directories(
  webgpu_backend
  PRIVATE $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)

target_link_libraries(webgpu_backend PRIVATE vulkan_schema executorch_core)

target_compile_options(webgpu_backend PRIVATE -fexceptions)

# Link with --whole-archive for static registration of backend + ops
executorch_target_link_options_shared_lib(webgpu_backend)

set_property(TARGET webgpu_backend PROPERTY CXX_STANDARD 17)

install(
  TARGETS webgpu_backend
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
