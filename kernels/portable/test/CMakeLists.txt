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

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(_kernels_portable_test_sources
    # So far we can't generate custom_ops.yaml in OSS so we can't build op
    # library with op_allclose. We disable the test for now.
    # "op_allclose_test.cpp"
    "op_div_test.cpp" "op_gelu_test.cpp" "op_mul_test.cpp"
)

add_executable(
  kernels_portable_test
  ${_kernels_portable_test_sources}
  ${EXECUTORCH_ROOT}/runtime/core/exec_aten/testing_util/tensor_util.cpp
)
target_link_libraries(
  kernels_portable_test
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
  executorch
  portable_kernels
  portable_ops_lib
)
target_include_directories(
  kernels_portable_test PRIVATE "${CMAKE_INSTALL_PREFIX}/include"
)
add_test(ExecuTorchTest kernels_portable_test)
