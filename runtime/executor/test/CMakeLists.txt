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
project(runtime_executor_test)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

et_cxx_test(
  runtime_executor_test SOURCES executor_test.cpp EXTRA_LIBS portable_ops_lib
  portable_kernels
)

et_cxx_test(
  allocation_failure_stress_test
  SOURCES
  allocation_failure_stress_test.cpp
  EXTRA_LIBS
  portable_ops_lib
  portable_kernels
  extension_data_loader
  extension_runner_util
)

et_cxx_test(
  method_test
  SOURCES
  method_test.cpp
  EXTRA_LIBS
  portable_ops_lib
  portable_kernels
  extension_data_loader
  extension_runner_util
)

# TODO(T191569140): Enable this test. et_cxx_test(method_meta_test SOURCES
# method_meta_test.cpp EXTRA_LIBS extension_data_loader)

# TODO(T191569140): Enable this test. et_cxx_test( program_test SOURCES
# program_test.cpp EXTRA_LIBS extension_data_loader )

# target_include_directories( program_test PRIVATE
# "${CMAKE_INSTALL_PREFIX}/schema/include"
# "${EXECUTORCH_ROOT}/third-party/flatbuffers/include" )

et_cxx_test(
  kernel_resolution_test SOURCES kernel_resolution_test.cpp EXTRA_LIBS
  extension_data_loader
)

et_cxx_test(
  kernel_integration_test SOURCES kernel_integration_test.cpp EXTRA_LIBS
  extension_data_loader extension_runner_util
)

# TODO(T191569140): Enable this test. et_cxx_test( backend_integration_test
# SOURCES backend_integration_test.cpp EXTRA_LIBS extension_data_loader
# extension_runner_util )

et_cxx_test(memory_manager_test SOURCES memory_manager_test.cpp)

et_cxx_test(
  tensor_parser_test
  SOURCES
  tensor_parser_test.cpp
  EXTRA_LIBS
  portable_ops_lib
  portable_kernels
  extension_data_loader
  extension_runner_util
)

target_include_directories(
  tensor_parser_test
  PRIVATE "${CMAKE_INSTALL_PREFIX}/schema/include"
          "${EXECUTORCH_ROOT}/third-party/flatbuffers/include"
)
