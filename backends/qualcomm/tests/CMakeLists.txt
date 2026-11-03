# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standalone Linux host tests: cmake -S backends/qualcomm/tests -B <build>
cmake_minimum_required(VERSION 3.24)
project(qnn_prerequisite_tests LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(
  EXECUTORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
)
set(EXECUTORCH_BUILD_QNN ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF CACHE BOOL "" FORCE)
add_subdirectory("${EXECUTORCH_ROOT}" executorch)
add_subdirectory("${EXECUTORCH_ROOT}/third-party/googletest" googletest)
enable_testing()

add_executable(
  qnn_io_buffer_allocator_test test_io_buffer_allocator.cpp
  ../llm/io_buffer_allocator.cpp
)
target_include_directories(
  qnn_io_buffer_allocator_test PRIVATE
  "${EXECUTORCH_ROOT}/.."
  "${EXECUTORCH_ROOT}/runtime/core/portable_type/c10"
  "${EXECUTORCH_ROOT}/third-party/flatbuffers/include"
  "${CMAKE_BINARY_DIR}/schema/include"
)
target_link_libraries(
  qnn_io_buffer_allocator_test PRIVATE executorch_core GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(qnn_io_buffer_allocator_test)
