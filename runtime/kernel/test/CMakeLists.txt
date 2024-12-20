# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

add_executable(operator_registry_test operator_registry_test.cpp)
target_link_libraries(
  operator_registry_test GTest::gtest GTest::gtest_main GTest::gmock executorch
)
target_include_directories(operator_registry_test PRIVATE ${EXECUTORCH_ROOT}/..)
add_test(operator_registry_test operator_registry_test)

add_executable(kernel_runtime_context_test kernel_runtime_context_test.cpp)
target_link_libraries(
  kernel_runtime_context_test GTest::gtest GTest::gtest_main GTest::gmock
  executorch
)
target_include_directories(
  kernel_runtime_context_test PRIVATE ${EXECUTORCH_ROOT}/..
)
add_test(kernel_runtime_context_test kernel_runtime_context_test)

add_executable(
  operator_registry_max_kernel_num_test
  operator_registry_max_kernel_num_test.cpp
  ../operator_registry.cpp
  ../../core/evalue.cpp
  ../../platform/abort.cpp
  ../../platform/log.cpp
  ../../platform/runtime.cpp
  ../../platform/default/posix.cpp
)
target_link_libraries(
  operator_registry_max_kernel_num_test GTest::gtest GTest::gtest_main
  GTest::gmock executorch
)
target_compile_definitions(
  operator_registry_max_kernel_num_test PRIVATE "-DMAX_KERNEL_NUM=1"
)
# TODO: This is currently not working!
# add_test(operator_registry_max_kernel_num_test operator_registry_max_kernel_num_test)

# TODO: Migrate kernel_double_registration_test and
# test_kernel_manual_registration. Make sure dtype selective build is working.
