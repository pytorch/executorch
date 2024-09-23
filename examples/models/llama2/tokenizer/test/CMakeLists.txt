# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
project(tokenizer_test)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_tokenizer_test_srcs
    test_tiktoken.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/tokenizer/tiktoken.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama_tiktoken.cpp
)

set(ENV{RESOURCES_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/resources)
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/third-party/abseil-cpp
  ${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp
)
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/third-party/re2
  ${CMAKE_CURRENT_BINARY_DIR}/re2
)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

et_cxx_test(tokenizer_test SOURCES ${_tokenizer_test_srcs} EXTRA_LIBS re2::re2)
target_include_directories(
  tokenizer_test
  PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/third-party/abseil-cpp
)
