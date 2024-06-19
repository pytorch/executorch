#!/usr/bin/env bash
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd ${script_dir}/../../.. && pwd)
ethos_u_root_dir=${et_root_dir}/examples/arm/ethos-u-scratch/ethos-u

toolchain_cmake=${et_root_dir}/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake
et_build_dir=${et_root_dir}/cmake-out
build_test_dir=${et_build_dir}/arm_semihosting_executor_runner
fvp_model=FVP_Corstone_SSE-300_Ethos-U55

# Build Arm Baremetal executor_runner in semihosting mode.
# Put in backends/arm/test/res to be used by unit tests.
function build_semihosting_executorch_runner() {
    cd ${et_root_dir}/examples/arm/executor_runner
    pwd
    mkdir -p ${build_test_dir}
    cmake -DCMAKE_TOOLCHAIN_FILE=${toolchain_cmake}          \
          -DTARGET_CPU=cortex-m55                            \
          -DSEMIHOSTING=ON                                   \
          -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${build_test_dir} \
          -B ${build_test_dir}                               \
          -DETHOS_SDK_PATH:PATH=${ethos_u_root_dir}          \
          -DET_DIR_PATH:PATH=${et_root_dir}                  \
          -DET_BUILD_DIR_PATH:PATH=${et_build_dir}           \
          -DPYTHON_EXECUTABLE=$(which python3)

    echo "[${FUNCNAME[0]}] Configured CMAKE"

    n=$(nproc)
    cmake --build ${build_test_dir} -- -j"$((n - 5))" arm_executor_runner
    echo "[${FUNCNAME[0]}] Generated baremetal elf file: with semihosting enabled"
    find ${build_test_dir} -name "arm_executor_runner"
}

build_semihosting_executorch_runner