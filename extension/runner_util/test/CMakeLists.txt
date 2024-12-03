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

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_test_srcs inputs_test.cpp)

et_cxx_test(
  extension_runner_util_test
  SOURCES
  ${_test_srcs}
  EXTRA_LIBS
  extension_data_loader
  extension_runner_util
  portable_kernels
  portable_ops_lib
)
set_property(
  TEST extension_runner_util_test
  PROPERTY ENVIRONMENT "ET_MODULE_ADD_PATH=${CMAKE_BINARY_DIR}/ModuleAdd.pte"
)
