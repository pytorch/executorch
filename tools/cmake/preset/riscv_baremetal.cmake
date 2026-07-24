# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Baremetal builds consume the build tree directly; mirror arm_baremetal so
# install rules stay invokable but write back into the build dir.
define_overridable_option(
  EXECUTORCH_BAREMETAL_SKIP_INSTALL
  "Skip emitting install/export rules when building bare-metal artifacts" BOOL
  ON
)

if(EXECUTORCH_BAREMETAL_SKIP_INSTALL)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")
  unset(CMAKE_SKIP_INSTALL_RULES CACHE)
  set(CMAKE_SKIP_INSTALL_RULES
      OFF
      CACHE
        BOOL
        "Retain install() rules so docs/scripts can keep calling --target install"
        FORCE
  )
endif()

set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER OFF)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR OFF)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
# BUNDLE_IO requires DEVTOOLS to provide the bundled_program lib.
set_overridable_option(EXECUTORCH_BUILD_DEVTOOLS ON)
set_overridable_option(EXECUTORCH_ENABLE_BUNDLE_IO ON)
set_overridable_option(EXECUTORCH_ENABLE_LOGGING ON)
# Freestanding target: no pthreadpool, no cpuinfo, no shared lib.
set_overridable_option(EXECUTORCH_BUILD_PTHREADPOOL OFF)
set_overridable_option(EXECUTORCH_BUILD_CPUINFO OFF)

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
