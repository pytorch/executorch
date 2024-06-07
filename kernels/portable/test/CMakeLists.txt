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
# This file builds portable-specific tests

cmake_minimum_required(VERSION 3.19)
project(kernels_portable_test)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_kernels_portable_test_sources
    # So far we can't generate custom_ops.yaml in OSS so we can't build op
    # library with op_allclose. We disable the test for now.
    # "op_allclose_test.cpp"
    "op_div_test.cpp" "op_gelu_test.cpp" "op_mul_test.cpp"
)

et_cxx_test(
  kernels_portable_test SOURCES ${_kernels_portable_test_sources} EXTRA_LIBS
  portable_kernels portable_ops_lib
)
target_include_directories(
  kernels_portable_test PRIVATE "${CMAKE_INSTALL_PREFIX}/include"
)
