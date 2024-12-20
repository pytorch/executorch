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
build_root_test_dir=${et_build_dir}/arm_semihosting_executor_runner
fvp_model=FVP_Corstone_SSE-300_Ethos-U55

# Build Arm Baremetal executor_runner in semihosting mode.
# Put in backends/arm/test/res to be used by unit tests.
function build_semihosting_executorch_runner() {
    target_board=$1
    system_config=$2
    build_test_dir=${build_root_test_dir}_${target_board}
    echo "[${FUNCNAME[0]}] Configuring ${target_board} with system config ${system_config}"
    if [[ ${target_board} == "corstone-300" ]]; then
        local target_cpu=cortex-m55
    elif [[ ${target_board} == "corstone-320" ]]; then
        local target_cpu=cortex-m85
    else
        echo "[${FUNCNAME[0]}] ERROR: Invalid target_board specified!"
        exit 1
    fi
    cd ${et_root_dir}/examples/arm/executor_runner
    pwd
    mkdir -p ${build_test_dir}
    cmake -DCMAKE_TOOLCHAIN_FILE=${toolchain_cmake}          \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo                  \
          -DTARGET_CPU=${target_cpu}                         \
          -DSEMIHOSTING=ON                                   \
          -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${build_test_dir} \
          -B ${build_test_dir}                               \
          -DETHOS_SDK_PATH:PATH=${ethos_u_root_dir}          \
          -DET_DIR_PATH:PATH=${et_root_dir}                  \
          -DET_BUILD_DIR_PATH:PATH=${et_build_dir}           \
          -DPYTHON_EXECUTABLE=$(which python3)               \
          -DSYSTEM_CONFIG=${system_config}
    echo "[${FUNCNAME[0]}] Configured CMAKE"

    n=$(nproc)
    cmake --build ${build_test_dir} -- -j"$((n - 5))" arm_executor_runner
    echo "[${FUNCNAME[0]}] Generated baremetal elf file: with semihosting enabled"
    find ${build_test_dir} -name "arm_executor_runner"
}

# Use most optimal system_configs for testing
build_semihosting_executorch_runner corstone-300 Ethos_U55_High_End_Embedded

build_semihosting_executorch_runner corstone-320 Ethos_U85_SYS_DRAM_Mid
