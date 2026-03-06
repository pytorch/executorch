# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

add_executable(
  file_backend_cache_test
  ${CMAKE_CURRENT_SOURCE_DIR}/file_backend_cache_test.cpp
)
target_link_libraries(
  file_backend_cache_test PRIVATE extension_backend_cache executorch_core
                                  GTest::gtest GTest::gtest_main
)
target_include_directories(
  file_backend_cache_test PRIVATE ${_common_include_directories}
)
add_test(NAME file_backend_cache_test COMMAND file_backend_cache_test)
