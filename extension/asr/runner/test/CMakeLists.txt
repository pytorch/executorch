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

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_test_srcs test_asr_runner.cpp)

# Add LSan stub for Apple platforms
if(APPLE)
  list(APPEND _test_srcs lsan_stub.cpp)
endif()

et_cxx_test(
  test_asr_runner SOURCES ${_test_srcs} EXTRA_LIBS extension_asr_runner
)
