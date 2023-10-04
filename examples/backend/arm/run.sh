#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

########
### Hardcoded constants
########
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(readlink -f ${script_dir}/../../../)
et_build_dir=${et_root_dir}/cmake-out
ethos_u_root_dir=$(readlink -f ${script_dir}/ethos-u/ethos-u)
ethos_u_build_dir=${ethos_u_root_dir}/core_platform/build
fvp_model=FVP_Corstone_SSE-300_Ethos-U55
toolchain_cmake=${ethos_u_root_dir}/core_platform/cmake/toolchain/arm-none-eabi-gcc.cmake
toolchain_cmake_executorch=${et_root_dir}/backends/arm/cmake/arm-none-eabi-gcc.cmake
_setup_msg="please refer to ${script_dir}/ethos-u-setup/setup.sh to properly install necessary tools."


# Generate eager mode results
# TODO

# Generate the PTE file
function generate_pte_file() {
    cd $et_root_dir
    python3 -m examples.export.export_example --model_name="softmax"
    local pte_file=$(readlink -f ./softmax.pte)
    [[ -f ${pte_file} ]] || { echo "Failed to generate a pte file - ${pte_file}"; exit 1; }
    echo "${pte_file}"
}

# build ExecuTorch Libraries
function build_executorch() {
    rm -rf "${et_build_dir}"
    mkdir "${et_build_dir}"
    cd "${et_build_dir}"
    cmake                                                      \
        -DBUCK2=/tmp/buck2                                     \
        -DFLATC_EXECUTABLE="$(which flatc)"                    \
        -DEXECUTORCH_BUILD_HOST_TARGETS=OFF                    \
        -DEXECUTORCH_BUILD_XNNPACK=OFF                         \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF                 \
        -DEXECUTORCH_BUILD_GFLAGS=OFF                          \
        -DCMAKE_SYSTEM_PROCESSOR=cortex-m55+nodsp+nofp         \
        -DETHOSU_TARGET_NPU_CONFIG=ethos-u55-128               \
        -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake_executorch}" \
        -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON                    \
        -DCMAKE_BUILD_TYPE=Release                             \
        -DEXECUTORCH_ENABLE_LOGGING_RELEASE_MODE=ON            \
        -DSELECT_OPS_LIST="aten::_softmax.out"                 \
        "${et_root_dir}"

    echo "[${FUNCNAME[0]}] Configured CMAKE"

    n=$(nproc)
    cmake --build . -j"$((n - 5))" -- VERBOSE=1
    echo "[${FUNCNAME[0]}] Generated static libraries for ExecuTorch:"
    find . -name "*.a" -exec ls -al {} \;
}

# build Arm Baremetal executor_runner
function build_executorch_runner() {
    [[ $# -ne 1 ]] && { echo "[${FUNCNAME[0]}] Expecting pte file as an argument got, $@"; exit 1; }
    local pte=${1}
    cd "${ethos_u_root_dir}"/core_platform
    cmake                                         \
        -DCMAKE_TOOLCHAIN_FILE=${toolchain_cmake_executorch} \
        -B build targets/corstone-300             \
        -DET_DIR_PATH:PATH=${et_root_dir}         \
        -DET_BUILD_DIR_PATH:PATH=${et_build_dir}  \
        -DET_PTE_FILE_PATH:PATH="${pte}"          \
        -DPYTHON_EXECUTABLE=$(which python3)
    echo "[${FUNCNAME[0]}] Configured CMAKE"

    n=$(nproc)
    cmake --build build -- -j"$((n - 5))" executor_runner executor_runner_delegate #VERBOSE=1
    echo "[${FUNCNAME[0]}] Generated baremetal elf file:"
    find . -name "executor_runner.elf"
}

# Execute the executor_runner on FVP Simulator
function run_fvp() {
    elf=$(find ${ethos_u_build_dir} -name "executor_runner.elf")
    [[ ! -f $elf ]] && { echo "[${FUNCNAME[0]}]: Unable to find executor_runner elf: ${elf}"; exit 1; }
    FVP_Corstone_SSE-300_Ethos-U55                          \
        -C ethosu.num_macs=128                              \
        -C mps3_board.visualisation.disable-visualisation=1 \
        -C mps3_board.telnetterminal0.start_telnet=0        \
        -C mps3_board.uart0.out_file='-'                    \
        -a "${elf}"                                         \
        --timelimit 5 || true
    echo "[${FUNCNAME[0]} Simulation complete, $?"
}

# Execute the executor_runner on FVP Simulator
function run_fvp_delegate() {
    elf=$(find ${ethos_u_build_dir} -name "executor_runner_delegate.elf")
    [[ ! -f $elf ]] && { echo "[${FUNCNAME[0]}]: Unable to find executor_runner_delegate elf: ${elf}"; exit 1; }
    FVP_Corstone_SSE-300_Ethos-U55                          \
        -C ethosu.num_macs=128                              \
        -C mps3_board.visualisation.disable-visualisation=1 \
        -C mps3_board.telnetterminal0.start_telnet=0        \
        -C mps3_board.uart0.out_file='-'                    \
        -a "${elf}"                                         \
        --timelimit 5 || true
    echo "[${FUNCNAME[0]} Simulation complete, $?"
}

#######
### Main
#######

# basic checks before we get started
hash ${fvp_model} \
    || { echo "Could not find ${fvp_model} on PATH, ${_setup_msg}"; exit 1; }

hash arm-none-eabi-gcc \
    || { echo "Could not find arm baremetal toolchain on PATH, ${_setup_msg}"; exit 1; }

[[ -f ${toolchain_cmake} ]] \
    || { echo "Could not find ${toolchain_cmake} file, ${_setup_msg}"; exit 1; }

[[ -f ${et_root_dir}/CMakeLists.txt ]] \
    || { echo "Executorch repo doesn't contain CMakeLists.txt file at root level"; exit 1; }

# get the pte
pte=$(generate_pte_file)

# build et
build_executorch

# build the et baremetal app
build_executorch_runner "${pte}"

# run the app
run_fvp 

# run the delegate app
run_fvp_delegate

exit $?
