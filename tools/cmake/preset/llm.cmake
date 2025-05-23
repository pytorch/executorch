# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Enable logging even when in release mode. We are building for desktop, where
# saving a few kB is less important than showing useful error information to
# users.
# keep sorted
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_CUSTOM ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set_overridable_option(EXECUTORCH_BUILD_XNNPACK ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_overridable_option(EXECUTORCH_BUILD_COREML ON)
  set_overridable_option(EXECUTORCH_BUILD_MPS ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Linux-specific code here
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "WIN32")
  # Windows or other OS-specific code here
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  # Android-specific code here
else()
  message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_NAME for LLM: ${CMAKE_SYSTEM_NAME}")
endif()
