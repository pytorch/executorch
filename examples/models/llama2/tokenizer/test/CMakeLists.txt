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
project(tokenizer_tests)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_tokenizer_test_srcs test_tiktoken.cpp test_bpe_tokenizer.cpp)

set(ENV{RESOURCES_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/resources)

et_cxx_test(
  tokenizer_tests
  SOURCES
  ${_tokenizer_test_srcs}
)
target_include_directories(
  tokenizer_tests PRIVATE ${CMAKE_INSTALL_PREFIX}/include
)
