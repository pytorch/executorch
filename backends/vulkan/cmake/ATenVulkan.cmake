# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT PYTORCH_PATH)
  set(PYTORCH_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch)
endif()

if(NOT VULKAN_THIRD_PARTY_PATH)
  set(VULKAN_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../third-party)
endif()

# Shader Codegen

# Trigger Shader code generation
set(USE_VULKAN ON)
include(${PYTORCH_PATH}/cmake/VulkanCodegen.cmake)

# Source paths and compile settings

set(ATEN_PATH ${PYTORCH_PATH}/aten/src)
set(ATEN_VULKAN_PATH ${ATEN_PATH}/ATen/native/vulkan)

set(VULKAN_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers/include)
set(VOLK_PATH ${VULKAN_THIRD_PARTY_PATH}/volk)
set(VMA_PATH ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator)

set(VULKAN_CXX_FLAGS "")
list(APPEND VULKAN_CXX_FLAGS "-DUSE_VULKAN_API")
list(APPEND VULKAN_CXX_FLAGS "-DUSE_VULKAN_WRAPPER")
list(APPEND VULKAN_CXX_FLAGS "-DUSE_VULKAN_VOLK")
list(APPEND VULKAN_CXX_FLAGS "-DVK_NO_PROTOTYPES")
list(APPEND VULKAN_CXX_FLAGS "-DVOLK_DEFAULT_VISIBILITY")

# vulkan_api_lib

file(GLOB vulkan_api_cpp ${ATEN_VULKAN_PATH}/api/*.cpp)

add_library(vulkan_api_lib STATIC ${vulkan_api_cpp} ${VOLK_PATH}/volk.c)

set(vulkan_api_headers)
list(APPEND vulkan_api_headers ${ATEN_PATH})
list(APPEND vulkan_api_headers ${VULKAN_HEADERS_PATH})
list(APPEND vulkan_api_headers ${VOLK_PATH})
list(APPEND vulkan_api_headers ${VMA_PATH})

target_include_directories(vulkan_api_lib PRIVATE ${vulkan_api_headers})

target_compile_options(vulkan_api_lib PRIVATE ${VULKAN_CXX_FLAGS})

# vulkan_shader_lib

file(GLOB vulkan_impl_cpp ${ATEN_VULKAN_PATH}/impl/*.cpp)

add_library(vulkan_shader_lib STATIC ${vulkan_impl_cpp} ${vulkan_generated_cpp})

list(APPEND vulkan_api_headers ${CMAKE_BINARY_DIR}/vulkan)

target_include_directories(vulkan_shader_lib PRIVATE ${vulkan_api_headers})

target_link_libraries(vulkan_shader_lib vulkan_api_lib)

target_compile_options(vulkan_shader_lib PRIVATE ${VULKAN_CXX_FLAGS})
