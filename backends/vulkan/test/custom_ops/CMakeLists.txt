# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Host-only tests that need neither a GPU nor the Vulkan backend library.
#
# ~~~
# cmake backends/vulkan/test/custom_ops -Bcmake-out/backends/vulkan/test/custom_ops
# cmake --build cmake-out/backends/vulkan/test/custom_ops
# ctest --test-dir cmake-out/backends/vulkan/test/custom_ops
# ~~~

cmake_minimum_required(VERSION 3.19)
project(vulkan_custom_ops_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
endif()

if(NOT TARGET GTest::gtest_main)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  add_subdirectory(
    ${EXECUTORCH_ROOT}/third-party/googletest
    ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL
  )
endif()

enable_testing()

set(VULKAN_OPS_PATH ${EXECUTORCH_ROOT}/backends/vulkan/runtime/graph/ops)

add_executable(
  vulkan_coopmat_family_test
  ${CMAKE_CURRENT_SOURCE_DIR}/coopmat_family_test.cpp
  ${VULKAN_OPS_PATH}/impl/gemm/linear/qw/CoopmatFamily.cpp
)
target_include_directories(
  vulkan_coopmat_family_test PRIVATE ${EXECUTORCH_ROOT}/..
)
target_compile_definitions(
  vulkan_coopmat_family_test
  PRIVATE COOPMAT_FAMILY_TEST_GLSL_DIR="${VULKAN_OPS_PATH}/glsl"
)
target_link_libraries(vulkan_coopmat_family_test PRIVATE GTest::gtest_main)
add_test(vulkan_coopmat_family_test vulkan_coopmat_family_test)
