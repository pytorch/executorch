# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

define_overridable_option(
  EXECUTORCH_BAREMETAL_SKIP_INSTALL
  "Skip emitting install/export rules when building bare-metal artifacts" BOOL
  ON
)

if(EXECUTORCH_BAREMETAL_SKIP_INSTALL)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")
  # Bare-metal builds consume the build tree directly, so skip generating
  # install rules to avoid exporting third-party SDK targets that live outside
  # the repo.
  set(CMAKE_SKIP_INSTALL_RULES ON)
endif()

set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR OFF)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER OFF)
set_overridable_option(EXECUTORCH_BUILD_ARM_BAREMETAL ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_CORTEX_M ON)
set_overridable_option(EXECUTORCH_ENABLE_LOGGING ON)

define_overridable_option(
  EXECUTORCH_BUILD_ARM_ETDUMP "Build etdump support for Arm" BOOL OFF
)

if("${EXECUTORCH_BUILD_ARM_ETDUMP}")
  set(EXECUTORCH_BUILD_DEVTOOLS ON)
  set(EXECUTORCH_ENABLE_EVENT_TRACER ON)
  set(FLATCC_ALLOW_WERROR OFF)
else()
  set(EXECUTORCH_ENABLE_EVENT_TRACER OFF)
endif()
