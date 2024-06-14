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
project(kernels_quantized_test)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_kernels_quantized_test_sources
    op_add_test.cpp
    op_choose_qparams_test.cpp
    op_dequantize_test.cpp
    op_embedding4b_test.cpp
    op_embedding_test.cpp
    op_mixed_linear_test.cpp
    op_mixed_mm_test.cpp
    op_quantize_test.cpp
)

et_cxx_test(
  kernels_quantized_test
  SOURCES
  ${_kernels_quantized_test_sources}
  EXTRA_LIBS
  quantized_kernels
  quantized_ops_lib
  portable_kernels
  portable_ops_lib
)
target_include_directories(
  kernels_quantized_test PRIVATE "${CMAKE_INSTALL_PREFIX}/include"
)
