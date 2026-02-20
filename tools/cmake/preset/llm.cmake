# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# keep sorted
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_ASR_RUNNER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
set_overridable_option(EXECUTORCH_BUILD_XNNPACK ON)

# Turn on the quantized and LLM kernels unless on Windows with MSVC build since
# they don't currently compile.
if(NOT ((CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL
                                                 "WIN32") AND MSVC)
)
  set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
  set_overridable_option(EXECUTORCH_BUILD_KERNELS_LLM ON)
else()
  if(NOT EXECUTORCH_BUILD_CUDA)
    message(
      WARNING
        "The llm custom kernels and the quantized kernels will not be built when using MSVC on Windows. "
        "If you need them (since you appear to be building for CPU) try building with -T ClangCL"
    )
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_overridable_option(EXECUTORCH_BUILD_COREML ON)
  set_overridable_option(EXECUTORCH_BUILD_MPS ON)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set_overridable_option(EXECUTORCH_BUILD_KERNELS_TORCHAO ON)
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Linux-specific code here
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL
                                               "WIN32"
)

elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  # Android-specific code here
else()
  message(
    FATAL_ERROR "Unsupported CMAKE_SYSTEM_NAME for LLM: ${CMAKE_SYSTEM_NAME}"
  )
endif()
