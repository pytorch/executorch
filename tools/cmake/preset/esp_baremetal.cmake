# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")
set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR OFF)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER OFF)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set_overridable_option(EXECUTORCH_ENABLE_LOGGING ON)
define_overridable_option(EXECUTORCH_ENABLE_EVENT_TRACER "Enable event tracer support" BOOL OFF)

if(EXECUTORCH_ENABLE_EVENT_TRACER)
  set(EXECUTORCH_BUILD_DEVTOOLS ON)
  set(FLATCC_ALLOW_WERROR OFF)
endif()
