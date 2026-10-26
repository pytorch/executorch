# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

find_package(Threads REQUIRED)
find_package(libzip CONFIG REQUIRED)
if(NOT TARGET nlohmann_json::nlohmann_json)
  find_package(nlohmann_json CONFIG REQUIRED)
endif()

add_subdirectory(runtime)

if(EXECUTORCH_BUILD_NATIVE_VULKAN)
  add_subdirectory(runtime/vulkan)
endif()

if(EXECUTORCH_BUILD_NATIVE_GGML)
  add_subdirectory(runtime/engine/ggml)
endif()

install(
  DIRECTORY runtime/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/native/runtime
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "test" EXCLUDE
  PATTERN "vulkan" EXCLUDE
  PATTERN "ggml" EXCLUDE
)

if(EXECUTORCH_BUILD_NATIVE_VISION_RUNNER)
  add_subdirectory(examples)
endif()

if(EXECUTORCH_BUILD_TESTS)
  add_subdirectory(test)
endif()
