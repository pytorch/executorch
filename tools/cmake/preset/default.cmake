# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(_is_build_type_release ON)
  set(_is_build_type_debug OFF)
else()
  set(_is_build_type_release OFF)
  set(_is_build_type_debug ON)
endif()

# MARK: - Overridable Options

define_overridable_option(
  EXECUTORCH_ENABLE_LOGGING
  "Build with ET_LOG_ENABLED"
  BOOL ${_is_build_type_debug}
)
define_overridable_option(
  EXECUTORCH_BUILD_COREML
  "Build the Core ML backend"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_FLATBUFFERS_MAX_ALIGNMENT
  "Exir lets users set the alignment of tensor data embedded in the flatbuffer, and some users need an alignment larger than the default, which is typically 32."
  STRING 1024
)
define_overridable_option(
  EXECUTORCH_PAL_DEFAULT
  "Which PAL default implementation to use. Choices: posix, minimal"
  STRING "posix"
)
define_overridable_option(
  EXECUTORCH_PAL_DEFAULT_FILE_PATH
  "PAL implementation file path"
  STRING "${PROJECT_SOURCE_DIR}/runtime/platform/default/${EXECUTORCH_PAL_DEFAULT}.cpp"
)
define_overridable_option(
  EXECUTORCH_LOG_LEVEL
  "Build with the given ET_MIN_LOG_LEVEL value"
  STRING "Info"
)
define_overridable_option(
  EXECUTORCH_ENABLE_PROGRAM_VERIFICATION
  "Build with ET_ENABLE_PROGRAM_VERIFICATION"
  BOOL ${_is_build_type_debug}
)
define_overridable_option(
  EXECUTORCH_ENABLE_EVENT_TRACER
  "Build with ET_EVENT_TRACER_ENABLED"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_OPTIMIZE_SIZE
  "Build executorch runtime optimizing for binary size"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_ARM_BAREMETAL
  "Build the Arm Baremetal flow for Cortex-M and Ethos-U"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_KERNELS_CUSTOM
  "Build the custom kernels"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT
  "Build the custom ops lib for AOT"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_DATA_LOADER
  "Build the Data Loader extension"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR
  "Build the Flat Tensor extension"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_LLM
  "Build the LLM extension"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_MODULE
  "Build the Module extension"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL
  "Build the Runner Util extension"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_TENSOR
  "Build the Tensor extension"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_TRAINING
  "Build the training extension"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_MPS
  "Build the MPS backend"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_NEURON
  "Build the backends/mediatek directory"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_OPENVINO
  "Build the Openvino backend"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_PYBIND
  "Build the Python Bindings"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_QNN
  "Build the Qualcomm backend"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_KERNELS_OPTIMIZED
  "Build the optimized kernels"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_KERNELS_QUANTIZED
  "Build the quantized kernels"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_DEVTOOLS
  "Build the ExecuTorch Developer Tools"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_TESTS
  "Build CMake-based unit tests"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_NNLIB_OPT
  "Build Cadence backend Hifi nnlib kernel"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_CADENCE_CPU_RUNNER
  "Build Cadence backend CPU runner"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_SIZE_TEST
  "Build the size test"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_XNNPACK
  "Build the XNNPACK backend"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_VULKAN
  "Build the Vulkan backend"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_PORTABLE_OPS
  "Build portable_ops library"
  BOOL ON
)
define_overridable_option(
  EXECUTORCH_USE_DL
  "Use libdl library"
  BOOL ON
)
define_overridable_option(
  EXECUTORCH_BUILD_CADENCE
  "Build the Cadence DSP backend"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_CORTEX_M
  "Build the Cortex-M backend"
  BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_GFLAGS
  "Build the gflags library."
  BOOL ON
)

if(EXECUTORCH_BUILD_ARM_BAREMETAL)
  set(_default_executorch_build_pthreadpool OFF)
  set(_default_executorch_build_cpuinfo OFF)
else()
  set(_default_executorch_build_pthreadpool ON)
  set(_default_executorch_build_cpuinfo ON)
endif()
define_overridable_option(
  EXECUTORCH_BUILD_PTHREADPOOL
  "Build pthreadpool library."
  BOOL ${_default_executorch_build_pthreadpool}
)
define_overridable_option(
  EXECUTORCH_BUILD_CPUINFO
  "Build cpuinfo library."
  BOOL ${_default_executorch_build_cpuinfo}
)

# TODO(jathu): move this to platform specific presets when created
set(_default_executorch_build_executor_runner ON)
if(APPLE AND "${SDK_NAME}" STREQUAL "iphoneos")
  set(_default_executorch_build_executor_runner OFF)
endif()
define_overridable_option(
  EXECUTORCH_BUILD_EXECUTOR_RUNNER
  "Build the executor_runner executable"
  BOOL ${_default_executorch_build_executor_runner}
)

# MARK: - Validations
# At this point all the options should be configured with their final value.

if(NOT EXISTS ${EXECUTORCH_PAL_DEFAULT_FILE_PATH})
  message(FATAL_ERROR "PAL default implementation (EXECUTORCH_PAL_DEFAULT=${EXECUTORCH_PAL_DEFAULT}) file not found: ${EXECUTORCH_PAL_DEFAULT_FILE_PATH}. Choices: posix, minimal")
endif()


string(TOLOWER "${EXECUTORCH_LOG_LEVEL}" _executorch_log_level_lower)
if(_executorch_log_level_lower STREQUAL "debug")
  set(ET_MIN_LOG_LEVEL Debug)
elseif(_executorch_log_level_lower STREQUAL "info")
  set(ET_MIN_LOG_LEVEL Info)
elseif(_executorch_log_level_lower STREQUAL "error")
  set(ET_MIN_LOG_LEVEL Error)
elseif(_executorch_log_level_lower STREQUAL "fatal")
  set(ET_MIN_LOG_LEVEL Fatal)
else()
  message(FATAL_ERROR "Unknown EXECUTORCH_LOG_LEVEL '${EXECUTORCH_LOG_LEVEL}'. Choices: Debug, Info, Error, Fatal")
endif()


if(EXECUTORCH_ENABLE_EVENT_TRACER)
  if(NOT EXECUTORCH_BUILD_DEVTOOLS)
    message(FATAL_ERROR "Use of 'EXECUTORCH_ENABLE_EVENT_TRACER' requires 'EXECUTORCH_BUILD_DEVTOOLS' to be enabled.")
  endif()
endif()


if(EXECUTORCH_BUILD_ARM_BAREMETAL)
  if(EXECUTORCH_BUILD_PTHREADPOOL)
    message(FATAL_ERROR "Cannot enable both EXECUTORCH_BUILD_PTHREADPOOL and EXECUTORCH_BUILD_ARM_BAREMETAL")
  elseif(EXECUTORCH_BUILD_CPUINFO)
    message(FATAL_ERROR "Cannot enable both EXECUTORCH_BUILD_CPUINFO and EXECUTORCH_BUILD_ARM_BAREMETAL")
  endif()
endif()
