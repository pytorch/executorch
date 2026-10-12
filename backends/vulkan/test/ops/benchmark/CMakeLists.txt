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
    set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
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
  set(PROTOTYPING_UTILS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/framework)
  set(PROTOTYPING_UTILS_CPP
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/cm_utils.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/config.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/conv2d_utils.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/device_info.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/registry.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/results.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/runner.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/test_case.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/value_spec.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/framework/weight_utils.cpp
  )

  # Prototyping shaders
  message(STATUS "shader stuff")
  set(PROTOTYPING_SHADERS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/test_ops/glsl)
  gen_vulkan_shader_lib_cpp(${PROTOTYPING_SHADERS_PATH})
  vulkan_shader_lib(prototyping_shaderlib ${generated_spv_cpp})
  target_compile_options(prototyping_shaderlib PRIVATE ${VULKAN_CXX_FLAGS})
  message(STATUS "done shader stuff")

  # Operator implementations library
  file(GLOB OPERATOR_IMPL_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/test_ops/impl/*.cpp
  )
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

  # gflags is built by the main ExecuTorch build but not installed, so find it
  # in that build tree, which build_and_run.sh places five levels above this
  # project's build directory.
  set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../../../third-party/gflags)
  find_package(gflags REQUIRED)

  file(GLOB_RECURSE ETVK_BENCH_CASES_CPP
       ${CMAKE_CURRENT_SOURCE_DIR}/cases/*.cpp
  )
  add_executable(
    etvk_bench ${CMAKE_CURRENT_SOURCE_DIR}/etvk_bench.cpp
               ${ETVK_BENCH_CASES_CPP} ${PROTOTYPING_UTILS_CPP}
  )
  target_include_directories(etvk_bench PRIVATE ${COMMON_INCLUDES})
  target_link_libraries(
    etvk_bench PRIVATE vulkan_backend executorch_core prototyping_shaderlib
                       operator_implementations gflags
  )
  target_compile_options(etvk_bench PRIVATE ${VULKAN_CXX_FLAGS})
  set_property(TARGET etvk_bench PROPERTY CXX_STANDARD 17)
endif()
