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
