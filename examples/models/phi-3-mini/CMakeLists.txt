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

cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0144 NEW)
project(phi_3_mini_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)

set(BUILD_TESTING OFF)
if(NOT TARGET extension_llm_runner)
  message(
    FATAL_ERROR
      "ExecuTorch must be installed with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER enabled."
  )
endif()

if(NOT TARGET gflags)
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/gflags
    ${CMAKE_BINARY_DIR}/gflags
  )
endif()

add_executable(phi_3_mini_runner main.cpp)

target_link_directories(phi_3_mini_runner PUBLIC ${_common_include_directories})

target_link_libraries(
  phi_3_mini_runner PUBLIC executorch optimized_native_cpu_ops_lib
                           xnnpack_backend gflags extension_llm_runner
)
