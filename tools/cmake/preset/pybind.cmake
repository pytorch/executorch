# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set_overridable_option(EXECUTORCH_BUILD_PYBIND ON)
set_overridable_option(EXECUTORCH_BUILD_CMSIS_NN_PYBINDS ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT ON)
# Enable logging and program verification even when in release mode. We are
# building for desktop, where saving a few kB is less important than showing
# useful error information to users.
set_overridable_option(EXECUTORCH_ENABLE_LOGGING ON)
set_overridable_option(EXECUTORCH_ENABLE_PROGRAM_VERIFICATION ON)
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

# Optional VGF enable for the default pybind/install flow. This is intentionally
# scoped to this preset rather than acting as a general environment-to-CMake
# override mechanism.
set(_executorch_pybind_enable_vgf OFF)
if(DEFINED ENV{EXECUTORCH_PYBIND_ENABLE_VGF})
  if("$ENV{EXECUTORCH_PYBIND_ENABLE_VGF}" STREQUAL "ON")
    set(_executorch_pybind_enable_vgf ON)
  else()
    set(_executorch_pybind_enable_vgf OFF)
  endif()
endif()

# TODO(larryliu0820): Temporarily disable building llm_runner for Windows wheel
# due to the issue of tokenizer file path length limitation.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # The wheel ships the profiler library and documents it as usable, so the
  # tracer has to be compiled in. Left off, every recording hook is preprocessed
  # away and a caller gets an empty trace with no error. Set per platform rather
  # than once above, because writing a trace with a debug buffer aborts the
  # interpreter on Windows, and a wheel that enables the hooks there hands that
  # crash to anyone who calls the profiling API.
  set_overridable_option(EXECUTORCH_ENABLE_EVENT_TRACER ON)
  # Same reason as on Linux: one shared runtime so a process has a single
  # backend registry, and a C++ consumer can link the wheel instead of building
  # from source. The Swift package remains the better fit for an application
  # bundle.
  set_overridable_option(EXECUTORCH_BUILD_SHARED ON)
  set_overridable_option(EXECUTORCH_BUILD_VGF ${_executorch_pybind_enable_vgf})
  set_overridable_option(EXECUTORCH_BUILD_COREML ON)
  set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TRAINING ON)
  set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON)
  set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM ON)
  # Both of these are Apple Silicon only. The TorchAO kernels build only for
  # aarch64, which is what TORCHAO_BUILD_CPU_AARCH64 selects; the Apple
  # framework build already ships them and this brings the wheel in line. MLX
  # additionally needs the Metal compiler (xcrun -sdk macosx metal), which comes
  # with Xcode and not with the Command Line Tools.
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set_overridable_option(EXECUTORCH_BUILD_KERNELS_TORCHAO ON)
    execute_process(
      COMMAND xcrun -sdk macosx --find metal
      RESULT_VARIABLE _metal_compiler_result
      OUTPUT_QUIET ERROR_QUIET
    )
    if(_metal_compiler_result EQUAL 0)
      set_overridable_option(EXECUTORCH_BUILD_MLX ON)
      set_overridable_option(ET_MLX_ENABLE_OP_LOGGING ON)
    else()
      message(
        STATUS
          "Metal compiler not found, disabling MLX backend. Install Xcode to enable MLX."
      )
    endif()
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_overridable_option(EXECUTORCH_ENABLE_EVENT_TRACER ON)

  set_overridable_option(EXECUTORCH_BUILD_VGF ${_executorch_pybind_enable_vgf})
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
  # Ship one shared runtime that both the pybind extension and standalone C++
  # consumers link, so a process has a single backend registry. Not set on
  # Windows, where the runtime has no export annotations for a DLL.
  set_overridable_option(EXECUTORCH_BUILD_SHARED ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL
                                               "WIN32"
)
  # Windows or other OS-specific code here
else()
  message(
    FATAL_ERROR "Unsupported CMAKE_SYSTEM_NAME for pybind: ${CMAKE_SYSTEM_NAME}"
  )
endif()

# Opt-in Vulkan backend for Linux/Windows wheels. Enabled ONLY when the build
# requests it via the EXECUTORCH_BUILD_VULKAN env var AND glslc (Vulkan SDK) is
# available to compile the shaders. This keeps the default wheel (and
# macOS/Android) byte-for-byte unchanged: GPU backends are opt-in rather than
# bundled into the universal wheel.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   OR CMAKE_SYSTEM_NAME STREQUAL "Windows"
   OR CMAKE_SYSTEM_NAME STREQUAL "WIN32"
)
  if(DEFINED ENV{EXECUTORCH_BUILD_VULKAN}
     AND NOT "$ENV{EXECUTORCH_BUILD_VULKAN}" STREQUAL "0"
     AND NOT "$ENV{EXECUTORCH_BUILD_VULKAN}" STREQUAL "OFF"
  )
    find_program(
      GLSLC_PATH glslc HINTS $ENV{VULKAN_SDK}/bin $ENV{VULKAN_SDK}/Bin
    )
    if(GLSLC_PATH)
      set_overridable_option(EXECUTORCH_BUILD_VULKAN ON)
      message(STATUS "Enabling Vulkan backend for wheel; glslc: ${GLSLC_PATH}")
    else()
      message(
        STATUS "EXECUTORCH_BUILD_VULKAN requested but glslc was not found; "
               "the Vulkan backend will not be included."
      )
    endif()
  endif()
endif()
