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
  # Bare-metal builds consume the build tree directly. Keep the install target
  # available (many docs/scripts still invoke it) but route the output back into
  # the build tree so nothing is exported outside the repo.
  unset(CMAKE_SKIP_INSTALL_RULES CACHE)
  set(CMAKE_SKIP_INSTALL_RULES OFF)
  set(CMAKE_SKIP_INSTALL_RULES
      OFF
      CACHE
        BOOL
        "Retain install() rules so docs/scripts can keep calling `--target install`"
        FORCE
  )
endif()

set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR OFF)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER OFF)
set_overridable_option(EXECUTORCH_BUILD_ARM_BAREMETAL ON)
# Bare-metal has no libdl (see CMakeLists.txt EXECUTORCH_USE_DL block).
set_overridable_option(EXECUTORCH_USE_DL OFF)
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
