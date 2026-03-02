# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set_overridable_option(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX ON)
set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)

set(_arm_ethosu_linux_c_flags_release "${CMAKE_C_FLAGS_RELEASE} -UNDEBUG")
set(CMAKE_C_FLAGS_RELEASE
    "${_arm_ethosu_linux_c_flags_release}"
    CACHE STRING "Avoid NDEBUG forward-decl mismatch in musl Release builds"
          FORCE
)
set(_arm_ethosu_linux_cxx_flags_release "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE
    "${_arm_ethosu_linux_cxx_flags_release}"
    CACHE STRING "Avoid NDEBUG forward-decl mismatch in musl Release builds"
          FORCE
)
