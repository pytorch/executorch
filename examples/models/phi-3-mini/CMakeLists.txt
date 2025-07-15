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

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
find_package(executorch CONFIG REQUIRED)

target_link_options_shared_lib(executorch)

set(BUILD_TESTING OFF)
add_subdirectory(${EXECUTORCH_ROOT}/extension/llm/runner ${CMAKE_BINARY_DIR}/../../../extension/llm/runner)

if(NOT TARGET gflags)
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/gflags
    ${CMAKE_BINARY_DIR}/gflags
  )
endif()

add_executable(
  phi_3_mini_runner
  main.cpp
)

target_link_libraries(
  phi_3_mini_runner PUBLIC executorch optimized_native_cpu_ops_lib xnnpack_backend gflags extension_llm_runner
)
