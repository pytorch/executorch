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
