
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set_overridable_option(EXECUTORCH_BUILD_PYBIND ON)
set_overridable_option(CMAKE_CXX_STANDARD_REQUIRED ON)
set_overridable_option(CMAKE_TOOLCHAIN_FILE "/home/zephyruser/executorch/examples/arm/ethos-u-setup/arm-zephyr-eabi-gcc.cmake")
set_overridable_option(CMAKE_SYSTEM_PROCESSOR cortex-m55)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED OFF)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT OFF)
set_overridable_option(EXECUTORCH_ENABLE_LOGGING ON)
set_overridable_option(EXECUTORCH_LOG_LEVEL Info)
set_overridable_option(EXECUTORCH_BUILD_XNNPACK ON)
set_overridable_option(EXECUTORCH_BUILD_PTHREADPOOL ON)
set_overridable_option(EXECUTORCH_BUILD_CPUINFO ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_CUSTOM ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER ON)
