# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(prototyping_shaders)

if(ANDROID)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif()

find_package(executorch CONFIG REQUIRED COMPONENTS vulkan_backend)

# Compile settings

set(VULKAN_CXX_FLAGS "-fexceptions")
list(APPEND VULKAN_CXX_FLAGS "-DUSE_VULKAN_WRAPPER")
list(APPEND VULKAN_CXX_FLAGS "-DUSE_VULKAN_VOLK")

message(STATUS "VULKAN_CXX_FLAGS: ${VULKAN_CXX_FLAGS}")

# Only build if Vulkan was compiled
if(TARGET vulkan_backend)
  if(NOT EXECUTORCH_ROOT)
    set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
  endif()

  if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python3)
  endif()

  # Include this file to access executorch_target_link_options_shared_lib
  include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
  include(${EXECUTORCH_ROOT}/backends/vulkan/cmake/ShaderLibrary.cmake)

  # Third party include paths
  set(VULKAN_THIRD_PARTY_PATH ${EXECUTORCH_ROOT}/backends/vulkan/third-party)
  set(VULKAN_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers/include)
  set(VOLK_PATH ${VULKAN_THIRD_PARTY_PATH}/volk)
  set(VMA_PATH ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator)

  set(COMMON_INCLUDES ${EXECUTORCH_ROOT}/.. ${VULKAN_HEADERS_PATH} ${VOLK_PATH}
                      ${VMA_PATH}
  )

  # Prototyping utility files
  set(PROTOTYPING_UTILS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR})
  set(PROTOTYPING_UTILS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

  # Prototyping shaders
  message(STATUS "shader stuff")
  set(PROTOTYPING_SHADERS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/glsl)
  gen_vulkan_shader_lib_cpp(${PROTOTYPING_SHADERS_PATH})
  vulkan_shader_lib(prototyping_shaderlib ${generated_spv_cpp})
  target_compile_options(prototyping_shaderlib PRIVATE ${VULKAN_CXX_FLAGS})
  message(STATUS "done shader stuff")

  # Operator implementations library
  file(GLOB OPERATOR_IMPL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/impl/*.cpp)
  add_library(operator_implementations STATIC ${OPERATOR_IMPL_SOURCES})
  target_include_directories(
    operator_implementations PRIVATE ${COMMON_INCLUDES}
  )
  target_link_libraries(
    operator_implementations PRIVATE vulkan_backend executorch_core
                                     prototyping_shaderlib
  )
  target_compile_options(operator_implementations PRIVATE ${VULKAN_CXX_FLAGS})
  set_property(TARGET operator_implementations PROPERTY CXX_STANDARD 17)

  executorch_target_link_options_shared_lib(vulkan_backend)
  executorch_target_link_options_shared_lib(operator_implementations)

  # Function to create operator prototype binaries
  function(add_operator_prototype OPERATOR_NAME)
    set(TARGET_NAME ${OPERATOR_NAME})
    set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${OPERATOR_NAME}.cpp)

    add_executable(${TARGET_NAME} ${SOURCE_FILE} ${PROTOTYPING_UTILS_CPP})
    target_include_directories(${TARGET_NAME} PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(
      ${TARGET_NAME} PRIVATE vulkan_backend executorch_core
                             prototyping_shaderlib operator_implementations
    )
    target_compile_options(${TARGET_NAME} PRIVATE ${VULKAN_CXX_FLAGS})
    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
  endfunction()

  # Define operator prototypes
  add_operator_prototype(add)
  add_operator_prototype(quantized_q8csw_linear)
endif()
