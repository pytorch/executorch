# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set_overridable_option(EXECUTORCH_BUILD_PYBIND ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT ON)
# Enable logging even when in release mode. We are building for desktop, where
# saving a few kB is less important than showing useful error information to
# users.
set_overridable_option(EXECUTORCH_ENABLE_LOGGING ON)
set_overridable_option(EXECUTORCH_LOG_LEVEL Info)
set_overridable_option(EXECUTORCH_BUILD_XNNPACK ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_LLM ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_LLM_AOT ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)

# TODO(larryliu0820): Temporarily disable building llm_runner for Windows
# wheel due to the issue of tokenizer file path length limitation.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_overridable_option(EXECUTORCH_BUILD_COREML ON)
  set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TRAINING ON)
  set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON)
  set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_overridable_option(EXECUTORCH_BUILD_COREML ON)
  set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TRAINING ON)
  set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON)
  set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL
                                               "WIN32"
)
  # Windows or other OS-specific code here
else()
  message(
    FATAL_ERROR "Unsupported CMAKE_SYSTEM_NAME for pybind: ${CMAKE_SYSTEM_NAME}"
  )
endif()
