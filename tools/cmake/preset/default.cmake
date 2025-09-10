# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_is_build_type_release ON)
  set(_is_build_type_debug OFF)
else()
  set(_is_build_type_release OFF)
  set(_is_build_type_debug ON)
endif()

# MARK: - Overridable Options

define_overridable_option(
  EXECUTORCH_ENABLE_LOGGING "Build with ET_LOG_ENABLED" BOOL
  ${_is_build_type_debug}
)
define_overridable_option(
  EXECUTORCH_BUILD_COREML "Build the Core ML backend" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_FLATBUFFERS_MAX_ALIGNMENT
  "Exir lets users set the alignment of tensor data embedded in the flatbuffer, and some users need an alignment larger than the default, which is typically 32."
  STRING
  1024
)
define_overridable_option(
  EXECUTORCH_PAL_DEFAULT
  "Which PAL default implementation to use. Choices: posix, minimal, android"
  STRING "posix"
)
define_overridable_option(
  EXECUTORCH_PAL_DEFAULT_FILE_PATH "PAL implementation file path" STRING
  "${PROJECT_SOURCE_DIR}/runtime/platform/default/${EXECUTORCH_PAL_DEFAULT}.cpp"
)
define_overridable_option(
  EXECUTORCH_LOG_LEVEL "Build with the given ET_MIN_LOG_LEVEL value" STRING
  "Info"
)
define_overridable_option(
  EXECUTORCH_ENABLE_PROGRAM_VERIFICATION
  "Build with ET_ENABLE_PROGRAM_VERIFICATION" BOOL ${_is_build_type_debug}
)
define_overridable_option(
  EXECUTORCH_ENABLE_EVENT_TRACER "Build with ET_EVENT_TRACER_ENABLED" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_OPTIMIZE_SIZE
  "Build executorch runtime optimizing for binary size" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_ARM_BAREMETAL
  "Build the Arm Baremetal flow for Cortex-M and Ethos-U" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_KERNELS_LLM "Build the custom kernels" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_KERNELS_LLM_AOT "Build the custom ops lib for AOT" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT
  "Build the optimized ops library for AOT export usage" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "Build the Data Loader extension" BOOL
  OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR "Build the Flat Tensor extension" BOOL
  OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_LLM "Build the LLM extension" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_LLM_APPLE "Build the LLM Apple extension" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER "Build the LLM runner extension" BOOL
  OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_MODULE "Build the Module extension" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_TENSOR "Build the Tensor extension" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_TRAINING "Build the training extension" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_APPLE "Build the Apple extension" BOOL OFF
)
define_overridable_option(EXECUTORCH_BUILD_MPS "Build the MPS backend" BOOL OFF)
define_overridable_option(
  EXECUTORCH_BUILD_NEURON "Build the backends/mediatek directory" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_OPENVINO "Build the Openvino backend" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_PYBIND "Build the Python Bindings" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_QNN "Build the Qualcomm backend" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_KERNELS_OPTIMIZED "Build the optimized kernels" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_KERNELS_QUANTIZED "Build the quantized kernels" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_DEVTOOLS "Build the ExecuTorch Developer Tools" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_TESTS "Build CMake-based unit tests" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_NNLIB_OPT "Build Cadence backend Hifi nnlib kernel" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_CADENCE_CPU_RUNNER "Build Cadence backend CPU runner" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_SIZE_TEST "Build the size test" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_XNNPACK "Build the XNNPACK backend" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_VULKAN "Build the Vulkan backend" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_PORTABLE_OPS "Build portable_ops library" BOOL ON
)
define_overridable_option(EXECUTORCH_USE_DL "Use libdl library" BOOL ON)
define_overridable_option(
  EXECUTORCH_BUILD_CADENCE "Build the Cadence DSP backend" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_CORTEX_M "Build the Cortex-M backend" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_VGF "Build the Arm VGF backend" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_COREML_BUILD_EXECUTOR_RUNNER "Build CoreML executor runner." BOOL
  OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_WASM "Build the ExecuTorch JavaScript API" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_BUILD_TOKENIZERS_WASM "Build the JavaScript Tokenizers API" BOOL
  OFF
)

if(EXECUTORCH_BUILD_ARM_BAREMETAL)
  set(_default_executorch_build_pthreadpool OFF)
  set(_default_executorch_build_cpuinfo OFF)
else()
  set(_default_executorch_build_pthreadpool ON)
  set(_default_executorch_build_cpuinfo ON)
endif()
define_overridable_option(
  EXECUTORCH_BUILD_PTHREADPOOL "Build pthreadpool library." BOOL
  ${_default_executorch_build_pthreadpool}
)
define_overridable_option(
  EXECUTORCH_BUILD_CPUINFO "Build cpuinfo library." BOOL
  ${_default_executorch_build_cpuinfo}
)

# TODO(jathu): move this to platform specific presets when created
set(_default_executorch_build_executor_runner ON)
if(APPLE AND "${SDK_NAME}" STREQUAL "iphoneos")
  set(_default_executorch_build_executor_runner OFF)
elseif(DEFINED EXECUTORCH_BUILD_PRESET_FILE)
  set(_default_executorch_build_executor_runner OFF)
endif()
define_overridable_option(
  EXECUTORCH_BUILD_EXECUTOR_RUNNER "Build the executor_runner executable" BOOL
  ${_default_executorch_build_executor_runner}
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL "Build the EValue util extension" BOOL
  ${_default_executorch_build_executor_runner}
)
define_overridable_option(
  EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL "Build the Runner Util extension" BOOL
  ${_default_executorch_build_executor_runner}
)

# NB: Enabling this will serialize execution of delegate instances Keeping this
# OFF by default to maintain existing behavior, to be revisited.
define_overridable_option(
  EXECUTORCH_XNNPACK_SHARED_WORKSPACE
  "Enable workspace sharing across different delegate instances" BOOL ON
)
# Keeping this OFF by default due to regressions in decode and model load with
# kleidi kernels
define_overridable_option(
  EXECUTORCH_XNNPACK_ENABLE_KLEIDI "Enable Arm Kleidi kernels" BOOL ON
)
# Turning this on cache weights between partitions and methods. If weights are
# shared across methods/partitions then this can reduce load time and memory
# usage
#
# Keeping this off maintains existing behavior. Turning this on serializes
# execution and initialization of delegates, to be revisited
define_overridable_option(
  EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE
  "Enable weights cache to cache and manage all packed weights" BOOL OFF
)
define_overridable_option(
  EXECUTORCH_USE_CPP_CODE_COVERAGE "Build with code coverage enabled" BOOL OFF
)

# Selective build options. These affect the executorch_kernels target.
define_overridable_option(
  EXECUTORCH_SELECT_OPS_YAML
  "Build the executorch_kernels target with YAML selective build config."
  STRING ""
)
define_overridable_option(
  EXECUTORCH_SELECT_OPS_LIST
  "Build the executorch_kernels target with a list of selected operators."
  STRING ""
)
define_overridable_option(
  EXECUTORCH_SELECT_OPS_MODEL
  "Build the executorch_kernels target with only operators from the given model .pte file."
  STRING
  ""
)
define_overridable_option(
  EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD
  "Build the executorch_kernels target with only operator implementations for selected data types."
  BOOL
  FALSE
)

# ------------------------------------------------------------------------------
# Validations
#
# At this point all the options should be configured with their final value.
# ------------------------------------------------------------------------------

check_required_options_on(
  IF_ON EXECUTORCH_ENABLE_EVENT_TRACER REQUIRES EXECUTORCH_BUILD_DEVTOOLS
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_EXECUTOR_RUNNER REQUIRES
  EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL
)
check_required_options_on(
  IF_ON EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR REQUIRES
  EXECUTORCH_BUILD_EXTENSION_DATA_LOADER
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_EXTENSION_LLM_APPLE REQUIRES
  EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER REQUIRES
  EXECUTORCH_BUILD_EXTENSION_LLM
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_EXTENSION_MODULE REQUIRES
  EXECUTORCH_BUILD_EXTENSION_DATA_LOADER EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_PYBIND REQUIRES EXECUTORCH_BUILD_EXTENSION_MODULE
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_KERNELS_LLM REQUIRES
  EXECUTORCH_BUILD_KERNELS_OPTIMIZED
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_KERNELS_LLM_AOT REQUIRES
  EXECUTORCH_BUILD_EXTENSION_TENSOR EXECUTORCH_BUILD_KERNELS_LLM
)

check_required_options_on(
  IF_ON
  EXECUTORCH_BUILD_EXTENSION_TRAINING
  REQUIRES
  EXECUTORCH_BUILD_EXTENSION_DATA_LOADER
  EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR
  EXECUTORCH_BUILD_EXTENSION_MODULE
  EXECUTORCH_BUILD_EXTENSION_TENSOR
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_TESTS REQUIRES EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR
)

check_required_options_on(
  IF_ON EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD REQUIRES
  EXECUTORCH_SELECT_OPS_MODEL
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_XNNPACK REQUIRES EXECUTORCH_BUILD_CPUINFO
  EXECUTORCH_BUILD_PTHREADPOOL
)

check_conflicting_options_on(
  IF_ON EXECUTORCH_BUILD_ARM_BAREMETAL CONFLICTS_WITH
  EXECUTORCH_BUILD_PTHREADPOOL EXECUTORCH_BUILD_CPUINFO
)

# Selective build specifiers are mutually exclusive.
check_conflicting_options_on(
  IF_ON EXECUTORCH_SELECT_OPS_YAML CONFLICTS_WITH EXECUTORCH_SELECT_OPS_LIST
  EXECUTORCH_SELECT_OPS_MODEL
)

check_conflicting_options_on(
  IF_ON EXECUTORCH_SELECT_OPS_LIST CONFLICTS_WITH EXECUTORCH_SELECT_OPS_MODEL
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_WASM REQUIRES EXECUTORCH_BUILD_EXTENSION_MODULE
  EXECUTORCH_BUILD_EXTENSION_TENSOR
)

check_required_options_on(
  IF_ON EXECUTORCH_BUILD_TOKENIZERS_WASM REQUIRES
  EXECUTORCH_BUILD_EXTENSION_LLM
)

if(NOT EXISTS ${EXECUTORCH_PAL_DEFAULT_FILE_PATH})
  message(
    FATAL_ERROR
      "PAL default implementation (EXECUTORCH_PAL_DEFAULT=${EXECUTORCH_PAL_DEFAULT}) file not found: ${EXECUTORCH_PAL_DEFAULT_FILE_PATH}. Choices: posix, minimal, android"
  )
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
  message(
    FATAL_ERROR
      "Unknown EXECUTORCH_LOG_LEVEL '${EXECUTORCH_LOG_LEVEL}'. Choices: Debug, Info, Error, Fatal"
  )
endif()
