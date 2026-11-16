# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.26)
project(native_sdk_test LANGUAGES CXX)
find_package(executorch_native CONFIG REQUIRED)
enable_testing()

set(NATIVE_TEST_PACKAGE
    ""
    CACHE FILEPATH "Path to the add.ptn test fixture"
)
option(NATIVE_TEST_VULKAN "Run the SDK test on a Vulkan device" OFF)
if(NOT EXISTS "${NATIVE_TEST_PACKAGE}")
  message(FATAL_ERROR "Generate add.ptn with make_fixture.py first")
endif()

add_executable(sdk_smoke sdk_smoke.cpp)
target_link_libraries(sdk_smoke PRIVATE executorch_native::runtime)
add_test(NAME native_sdk_ggml COMMAND sdk_smoke "${NATIVE_TEST_PACKAGE}" ggml)
if(NATIVE_TEST_VULKAN)
  add_test(NAME native_sdk_vulkan COMMAND sdk_smoke "${NATIVE_TEST_PACKAGE}"
                                          vulkan
  )
endif()
