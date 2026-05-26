# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_DEVTOOLS ON)
set_overridable_option(EXECUTORCH_ENABLE_BUNDLE_IO ON)
set_overridable_option(EXECUTORCH_ENABLE_LOGGING ON)

define_overridable_option(
  EXECUTORCH_BUILD_RISCV_ETDUMP "Build etdump support for RISC-V" BOOL OFF
)

if("${EXECUTORCH_BUILD_RISCV_ETDUMP}")
  set(EXECUTORCH_BUILD_DEVTOOLS ON)
  set(EXECUTORCH_ENABLE_EVENT_TRACER ON)
  set(FLATCC_ALLOW_WERROR OFF)
else()
  set(EXECUTORCH_ENABLE_EVENT_TRACER OFF)
endif()

if(EXECUTORCH_BUILD_XNNPACK)
  if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_C_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR "XNNPACK requires GCC 14+ on riscv64")
  endif()
elseif(NOT DEFINED EXECUTORCH_BUILD_XNNPACK)
  if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL
                                 14
  )
    set(EXECUTORCH_BUILD_XNNPACK ON)
  else()
    message(NOTICE "XNNPACK requires GCC 14+ on riscv64")
  endif()
endif()
