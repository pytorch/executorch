# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
# The targets in this file will be built if EXECUTORCH_BUILD_VULKAN is ON

if(NOT PYTORCH_PATH)
  set(PYTORCH_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch)
endif()

if(NOT VULKAN_THIRD_PARTY_PATH)
  set(VULKAN_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../third-party)
endif()

# Shader Codegen

# Trigger Shader code generation
set(USE_VULKAN ON)
set(VULKAN_CODEGEN_CMAKE_PATH ${PYTORCH_PATH}/cmake/VulkanCodegen.cmake)
if(NOT EXISTS ${VULKAN_CODEGEN_CMAKE_PATH})
  message(
    FATAL_ERROR
      "Cannot perform SPIR-V codegen because " ${VULKAN_CODEGEN_CMAKE_PATH}
      " does not exist. Please make sure that submodules are initialized"
      " and updated.")
endif()
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

file(GLOB VULKAN_API_CPP ${ATEN_VULKAN_PATH}/api/*.cpp)

add_library(vulkan_api_lib STATIC ${VULKAN_API_CPP} ${VOLK_PATH}/volk.c)

set(VULKAN_API_HEADERS)
list(APPEND VULKAN_API_HEADERS ${ATEN_PATH})
list(APPEND VULKAN_API_HEADERS ${VULKAN_HEADERS_PATH})
list(APPEND VULKAN_API_HEADERS ${VOLK_PATH})
list(APPEND VULKAN_API_HEADERS ${VMA_PATH})

target_include_directories(vulkan_api_lib PRIVATE ${VULKAN_API_HEADERS})

target_compile_options(vulkan_api_lib PRIVATE ${VULKAN_CXX_FLAGS})

# vulkan_shader_lib

file(GLOB VULKAN_IMPL_CPP ${ATEN_VULKAN_PATH}/impl/*.cpp)

add_library(vulkan_shader_lib STATIC ${VULKAN_IMPL_CPP} ${vulkan_generated_cpp})

list(APPEND VULKAN_API_HEADERS ${CMAKE_BINARY_DIR}/vulkan)

target_include_directories(vulkan_shader_lib PRIVATE ${VULKAN_API_HEADERS})

target_link_libraries(vulkan_shader_lib vulkan_api_lib)

target_compile_options(vulkan_shader_lib PRIVATE ${VULKAN_CXX_FLAGS})
