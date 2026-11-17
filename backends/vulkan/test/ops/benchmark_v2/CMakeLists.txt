# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(etvk_ops_bench)

if(ANDROID)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif()

find_package(executorch CONFIG REQUIRED COMPONENTS vulkan_backend)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

# Include this file to access executorch_target_link_options_shared_lib
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/backends/vulkan/cmake/ShaderLibrary.cmake)

set(VULKAN_CXX_FLAGS -fexceptions -DUSE_VULKAN_WRAPPER -DUSE_VULKAN_VOLK)

set(VULKAN_THIRD_PARTY_PATH ${EXECUTORCH_ROOT}/backends/vulkan/third-party)
set(COMMON_INCLUDES
    ${EXECUTORCH_ROOT}/..
    ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers/include
    ${VULKAN_THIRD_PARTY_PATH}/volk
    ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator
    ${EXECUTORCH_ROOT}/third-party/json/include
)

gen_vulkan_shader_lib_cpp(
  ${CMAKE_CURRENT_SOURCE_DIR}/operators/gemm/linear/qw/q4gsw/glsl
)
vulkan_shader_lib(prototype_shaderlib ${generated_spv_cpp})

# Compiling every source into the executable keeps the static route, reference,
# throughput and check registrations that would be dropped from an archive.
file(
  GLOB_RECURSE ETVK_OPS_BENCH_SOURCES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  framework/*.cpp operators/*.cpp
)
list(FILTER ETVK_OPS_BENCH_SOURCES EXCLUDE REGEX "(^|/)test/|Test\\.cpp$")

# gflags is built by the main ExecuTorch build but not installed, so find it in
# that build tree, which build.sh places five levels above this project's build
# directory.
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../../../third-party/gflags)
find_package(gflags REQUIRED)

add_executable(
  etvk_ops_bench ${CMAKE_CURRENT_SOURCE_DIR}/etvk_ops_bench.cpp
                 ${ETVK_OPS_BENCH_SOURCES}
)
target_include_directories(etvk_ops_bench PRIVATE ${COMMON_INCLUDES})
target_link_libraries(
  etvk_ops_bench PRIVATE vulkan_backend executorch_core prototype_shaderlib
                         gflags
)
target_compile_options(etvk_ops_bench PRIVATE ${VULKAN_CXX_FLAGS})
set_property(TARGET etvk_ops_bench PROPERTY CXX_STANDARD 20)

executorch_target_link_options_shared_lib(vulkan_backend)
