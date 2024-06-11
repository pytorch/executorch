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
project(sdk_etdump_tests)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_test_srcs etdump_test.cpp)

et_cxx_test(
  sdk_etdump_tests
  SOURCES
  ${_test_srcs}
  EXTRA_LIBS
  bundled_program
  etdump
  flatccrt_d
)
target_include_directories(
  sdk_etdump_tests PRIVATE ${CMAKE_INSTALL_PREFIX}/sdk/include
                           ${EXECUTORCH_ROOT}/third-party/flatcc/include
)
