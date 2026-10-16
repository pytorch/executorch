# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT TARGET gflags AND NOT TARGET gflags::gflags)
  find_package(gflags REQUIRED)
endif()
if(TARGET gflags::gflags)
  set(_native_runner_gflags gflags::gflags)
else()
  set(_native_runner_gflags gflags)
endif()

add_subdirectory(runner_utils)
if(EXECUTORCH_BUILD_NATIVE_VISION_RUNNER)
  if(NOT EXECUTORCH_BUILD_NATIVE_VULKAN)
    message(FATAL_ERROR "The native vision runner requires Native Vulkan")
  endif()
  add_subdirectory(vision)
endif()
if(EXECUTORCH_BUILD_NATIVE_LLM_RUNNER)
  add_subdirectory(hf/runtime)
endif()
