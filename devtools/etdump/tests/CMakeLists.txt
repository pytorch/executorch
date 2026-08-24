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

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_test_srcs etdump_test.cpp)

# etdump_test.cpp includes etdump_filter.h, which needs re2, and the devtools
# build does not pull re2 in.
if(TARGET re2::re2)
  et_cxx_test(
    sdk_etdump_tests
    SOURCES
    ${_test_srcs}
    EXTRA_LIBS
    bundled_program
    etdump
    flatccrt
  )
  target_include_directories(
    sdk_etdump_tests PRIVATE ${CMAKE_INSTALL_PREFIX}/sdk/include
                             ${EXECUTORCH_ROOT}/third-party/flatcc/include
  )
endif()

et_cxx_test(
  etdump_device_test SOURCES etdump_device_test.cpp EXTRA_LIBS etdump flatccrt
)

et_cxx_test(
  etdump_device_no_allocator_test SOURCES etdump_device_no_allocator_test.cpp
  EXTRA_LIBS etdump flatccrt
)
