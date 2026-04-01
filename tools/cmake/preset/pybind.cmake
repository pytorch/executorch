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
set_overridable_option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON)
set_overridable_option(EXECUTORCH_BUILD_WHEEL_DO_NOT_USE ON)

# TODO(larryliu0820): Temporarily disable building llm_runner for Windows wheel
# due to the issue of tokenizer file path length limitation.
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
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|i.86)$")
    # Auto-enable QNN on Linux x86 when the SDK is available. - QNN_SDK_ROOT set
    # explicitly → always enable - GitHub Actions CI → skip (avoids flaky 1.3GB
    # downloads) - Otherwise → probe the download server; skip gracefully when
    # unreachable (e.g. devvms without proxy configured)
    if(DEFINED QNN_SDK_ROOT OR DEFINED ENV{QNN_SDK_ROOT})
      set_overridable_option(EXECUTORCH_BUILD_QNN ON)
    elseif("$ENV{GITHUB_ACTIONS}" STREQUAL "true")
      message(STATUS "GitHub Actions CI detected: skipping QNN auto-download. "
                     "Set QNN_SDK_ROOT or -DEXECUTORCH_BUILD_QNN=ON to enable."
      )
    else()
      execute_process(
        COMMAND
          ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_LIST_DIR}/../../../backends/qualcomm/scripts/download_qnn_sdk.py
          --check
        RESULT_VARIABLE _qnn_available
        OUTPUT_QUIET ERROR_QUIET
        TIMEOUT 10
      )
      if(_qnn_available EQUAL 0)
        set_overridable_option(EXECUTORCH_BUILD_QNN ON)
      else()
        message(
          STATUS "QNN SDK not cached and download server unreachable. "
                 "Skipping QNN backend. Set QNN_SDK_ROOT or use "
                 "-DEXECUTORCH_BUILD_QNN=ON with network access to enable."
        )
      endif()
    endif()
  endif()
  set_overridable_option(EXECUTORCH_BUILD_OPENVINO OFF)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL
                                               "WIN32"
)
  # Windows or other OS-specific code here
else()
  message(
    FATAL_ERROR "Unsupported CMAKE_SYSTEM_NAME for pybind: ${CMAKE_SYSTEM_NAME}"
  )
endif()
