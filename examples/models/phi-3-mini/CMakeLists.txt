# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
project(phi_3_mini_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

# Set options for executorch build.
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ${CMAKE_BINARY_DIR}/../../..
)
if(NOT TARGET gflags)
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/gflags
    ${CMAKE_BINARY_DIR}/gflags
  )
endif()

add_executable(
  phi_3_mini_runner
  main.cpp runner.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../extension/llm/sampler/sampler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../extension/llm/tokenizer/bpe_tokenizer.cpp
)
target_include_directories(
  phi_3_mini_runner
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/gflags/src
)
target_link_libraries(
  phi_3_mini_runner PRIVATE executorch extension_module_static extension_tensor
                            optimized_native_cpu_ops_lib xnnpack_backend gflags
)
