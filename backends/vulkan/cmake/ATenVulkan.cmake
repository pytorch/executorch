# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format --first-comment-is-literal=True -i ATenVulkan.cmake
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

# Find GLSL compiler executable

if(ANDROID)
  if(NOT ANDROID_NDK)
    message(FATAL_ERROR "ANDROID_NDK not set")
  endif()

  set(GLSLC_PATH
      "${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}/glslc")
else()
  find_program(
    GLSLC_PATH glslc
    PATHS ENV VULKAN_SDK
    PATHS "$ENV{VULKAN_SDK}/${CMAKE_HOST_SYSTEM_PROCESSOR}/bin"
    PATHS "$ENV{VULKAN_SDK}/bin")

  if(NOT GLSLC_PATH)
    message(FATAL_ERROR "USE_VULKAN glslc not found")
  endif()
endif()

# Required to enable linking with --whole-archive
include(${EXECUTORCH_ROOT}/build/Utils.cmake)

# Convenience macro to create a shader library

macro(vulkan_shader_library SHADERS_PATH LIBRARY_NAME)
  set(VULKAN_SHADERGEN_ENV "")
  set(VULKAN_SHADERGEN_OUT_PATH ${CMAKE_BINARY_DIR}/${LIBRARY_NAME})

  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" ${PYTORCH_PATH}/tools/gen_vulkan_spv.py --glsl-path
      ${SHADERS_PATH} --output-path ${VULKAN_SHADERGEN_OUT_PATH}
      --glslc-path=${GLSLC_PATH} --tmp-dir-path=${VULKAN_SHADERGEN_OUT_PATH}
      --env ${VULKAN_GEN_ARG_ENV}
    RESULT_VARIABLE error_code)
  set(ENV{PYTHONPATH} ${PYTHONPATH})

  set(vulkan_generated_cpp ${VULKAN_SHADERGEN_OUT_PATH}/spv.cpp)

  add_library(${LIBRARY_NAME} STATIC ${vulkan_generated_cpp})
  target_include_directories(${LIBRARY_NAME} PRIVATE ${COMMON_INCLUDES})
  target_link_libraries(${LIBRARY_NAME} vulkan_api_lib)
  target_compile_options(${LIBRARY_NAME} PRIVATE ${VULKAN_CXX_FLAGS})
  # Link this library with --whole-archive due to dynamic shader registrations
  target_link_options_shared_lib(${LIBRARY_NAME})

  unset(VULKAN_SHADERGEN_ENV)
  unset(VULKAN_SHADERGEN_OUT_PATH)
endmacro()
