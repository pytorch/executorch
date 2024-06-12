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
project(runtime_kernel_test)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

add_executable(operator_registry_test operator_registry_test.cpp)
target_link_libraries(
  operator_registry_test GTest::gtest GTest::gtest_main GTest::gmock executorch
)
target_include_directories(operator_registry_test PRIVATE ${EXECUTORCH_ROOT}/..)
add_test(ExecuTorchTest operator_registry_test)


add_executable(kernel_runtime_context_test kernel_runtime_context_test.cpp)
target_link_libraries(
  kernel_runtime_context_test GTest::gtest GTest::gtest_main GTest::gmock executorch
)
target_include_directories(kernel_runtime_context_test PRIVATE ${EXECUTORCH_ROOT}/..)
add_test(ExecuTorchTest kernel_runtime_context_test)


add_executable(operator_registry_max_kernel_num_test operator_registry_max_kernel_num_test.cpp ../operator_registry.cpp
 ../../core/evalue.cpp ../../platform/abort.cpp ../../platform/log.cpp ../../platform/runtime.cpp ../../platform/target/Posix.cpp)
target_link_libraries(
  operator_registry_max_kernel_num_test GTest::gtest GTest::gtest_main GTest::gmock
)
target_compile_definitions(operator_registry_max_kernel_num_test PRIVATE "-DMAX_KERNEL_NUM=1")
target_include_directories(operator_registry_max_kernel_num_test PRIVATE ${EXECUTORCH_ROOT}/..)
add_test(ExecuTorchTest operator_registry_max_kernel_num_test)


include(${EXECUTORCH_ROOT}/build/Codegen.cmake)
gen_selected_ops(LIB_NAME "specialized_ops_lib" OPS_SCHEMA_YAML "${CMAKE_CURRENT_LIST_DIR}/functions.yaml")
generate_bindings_for_kernels(
  LIB_NAME "specialized_ops_lib" FUNCTIONS_YAML "${CMAKE_CURRENT_LIST_DIR}/functions.yaml"
)
gen_operators_lib(
  LIB_NAME "specialized_ops_lib" KERNEL_LIBS portable_kernels DEPS executorch
)
et_cxx_test(runtime_double_registration_test SOURCES kernel_double_registration_test.cpp)
target_link_libraries(
  runtime_double_registration_test GTest::gtest GTest::gtest_main GTest::gmock executorch
)
target_include_directories(runtime_double_registration_test PRIVATE ${EXECUTORCH_ROOT}/..)
add_test(ExecuTorchTest runtime_double_registration_test)

# add_executable(test_kernel_manual_registration_test test_kernel_manual_registration.cpp)
# target_link_libraries(
#   test_kernel_manual_registration_test GTest::gtest GTest::gtest_main GTest::gmock executorch portable_kernels portable_ops_lib
# )
# target_include_directories(test_kernel_manual_registration_test PRIVATE ${EXECUTORCH_ROOT}/..)
# add_test(ExecuTorchTest test_kernel_manual_registration_test)
