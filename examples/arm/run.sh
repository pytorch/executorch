#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

if [[ "${1:-'.'}" == "-h" || "${#}" -gt 2 ]]; then
    echo "Usage: $(basename $0) [path-to-a-scratch-dir]"
    echo "Supplied args: $*"
    exit 1
fi

########
### Hardcoded constants
########
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# Ethos-u
root_dir=${1:-"${script_dir}/ethos-u-scratch"}
root_dir=$(realpath ${root_dir})

ethos_u_root_dir="$(cd ${root_dir}/ethos-u && pwd)"
ethos_u_build_dir=${ethos_u_root_dir}/core_platform/build
setup_path_script=${root_dir}/setup_path.sh

# Executorch
et_root_dir=$(cd ${script_dir}/../.. && pwd)
et_build_dir=${et_root_dir}/cmake-out

fvp_model=FVP_Corstone_SSE-300_Ethos-U55
toolchain_cmake=${script_dir}/ethos-u-setup/arm-none-eabi-gcc.cmake
_setup_msg="please refer to ${script_dir}/ethos-u-setup/setup.sh to properly install necessary tools."

# Generate a pte file
function generate_pte_file() {
    [[ $# -ne 2 ]] && { echo "[${FUNCNAME[0]}]" "Expecting model and delegate flag, got, $*"; exit 1; }
    local model=${1}
    local delegate=${2}

    local model_filename=${model}.pte
    if [[ "${delegate}" == *"--delegate"* ]]; then
        model_filename=${model}_arm_delegate.pte
    fi
    cd $et_root_dir

    local pte_file
    pte_file=$(realpath ${model_filename})
    rm -f "${pte_file}"

    # We are using the aot_lib from build_quantization_aot_lib below
    SO_LIB=$(find cmake-out-aot-lib -name libquantized_ops_aot_lib.so)

    python3 -m examples.arm.aot_arm_compiler --model_name="${model}" ${delegate} --so_library="$SO_LIB" 1>&2
    [[ -f ${pte_file} ]] || { echo "Failed to generate a pte file - ${pte_file}"; exit 1; }
    echo "${pte_file}"
}

# Build .so library to register quant ops with AoT flow
function build_quantization_aot_lib()
{
    SITE_PACKAGES="$(python3 -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
    CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch"

    cd $et_root_dir
    mkdir -p cmake-out-aot-lib
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_XNNPACK=OFF \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
        -DPYTHON_EXECUTABLE=python3 \
        -Bcmake-out-aot-lib \
        "${et_root_dir}"

    n=$(nproc)
    cmake --build cmake-out-aot-lib -j"$((n - 5))" -- quantized_ops_aot_lib
}


# build ExecuTorch Libraries
function build_executorch() {
    set -x

    [[ -d "${et_build_dir}" ]] \
        && echo "[${FUNCNAME[0]}] Warn: using already existing build-dir for executorch: ${et_build_dir}!!"
    mkdir -p "${et_build_dir}"

    cd "${et_root_dir}"
    cmake                                                 \
        -DCMAKE_INSTALL_PREFIX=${et_build_dir}            \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF            \
        -DCMAKE_BUILD_TYPE=Release                        \
        -DEXECUTORCH_ENABLE_LOGGING=ON                    \
        -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON               \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON                   \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON       \
        -DFLATC_EXECUTABLE="$(which flatc)"               \
        -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"       \
        -B${et_build_dir}                                 \
        "${et_root_dir}"

    echo "[${FUNCNAME[0]}] Configured CMAKE"

    n=$(nproc)
    cmake --build ${et_build_dir} -j"$((n - 5))" --target install --config Release

    cmake                                                 \
        -DCMAKE_INSTALL_PREFIX=${et_build_dir}            \
        -DCMAKE_BUILD_TYPE=Release                        \
        -DEXECUTORCH_SELECT_OPS_LIST="aten::_softmax.out" \
        -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON               \
        -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"       \
        -B"${et_build_dir}"/examples/arm                  \
        "${et_root_dir}"/examples/arm
    cmake --build ${et_build_dir}/examples/arm -- -j"$((n - 5))"

    set +x

    cd "${et_build_dir}"
    echo "[${FUNCNAME[0]}] Generated static libraries for ExecuTorch:"
    find . -name "*.a" -exec ls -al {} \;
}

# build Arm Baremetal executor_runner
function build_executorch_runner() {
    echo "[${FUNCNAME[0]}] Generating ExecuTorch libraries"
    [[ $# -ne 1 ]] && { echo "[${FUNCNAME[0]}]" "Expecting a single pte file as argument got, $*"; exit 1; }
    local pte=${1}
    cd ${script_dir}/executor_runner
    cmake -DCMAKE_TOOLCHAIN_FILE=${toolchain_cmake} \
	  -DTARGET_CPU=cortex-m55 \
	  -B cmake-out \
	  -DETHOS_SDK_PATH:PATH=${ethos_u_root_dir} \
	  -DET_DIR_PATH:PATH=${et_root_dir}         \
	  -DET_BUILD_DIR_PATH:PATH=${et_build_dir}  \
	  -DET_PTE_FILE_PATH:PATH="${pte}"          \
	  -DPYTHON_EXECUTABLE=$(which python3)
    echo "[${FUNCNAME[0]}] Configured CMAKE"

    n=$(nproc)
    cmake --build cmake-out -- -j"$((n - 5))" arm_executor_runner
    echo "[${FUNCNAME[0]}] Generated baremetal elf file:"
    find cmake-out -name "arm_executor_runner"
}

# Execute the executor_runner on FVP Simulator
function run_fvp() {
    [[ $# -ne 1 ]] && { echo "[${FUNCNAME[0]}]" "Expexted elf binary name, got $*"; exit 1; }
    local elf_name=${1}
    elf=$(find ${script_dir}/executor_runner -name "${elf_name}")
    [[ ! -f $elf ]] && { echo "[${FUNCNAME[0]}]: Unable to find executor_runner elf: ${elf}"; exit 1; }
    FVP_Corstone_SSE-300_Ethos-U55                          \
        -C cpu0.CFGITCMSZ=11                                \
        -C ethosu.num_macs=128                              \
        -C mps3_board.visualisation.disable-visualisation=1 \
        -C mps3_board.telnetterminal0.start_telnet=0        \
        -C mps3_board.uart0.out_file='-'                    \
        -C mps3_board.uart0.shutdown_on_eot=1               \
        -a "${elf}"                                         \
        --timelimit 120 || true # seconds
    echo "[${FUNCNAME[0]} Simulation complete, $?"
}

#######
### Main
#######
# Source the tools
# This should be prepared by the setup.sh
[[ -f ${setup_path_script} ]] \
    || { echo "Missing ${setup_path_script}. ${_setup_msg}"; exit 1; }
source ${root_dir}/setup_path.sh

# basic checks before we get started
hash ${fvp_model} \
    || { echo "Could not find ${fvp_model} on PATH, ${_setup_msg}"; exit 1; }

hash arm-none-eabi-gcc \
    || { echo "Could not find arm baremetal toolchain on PATH, ${_setup_msg}"; exit 1; }

[[ -f ${toolchain_cmake} ]] \
    || { echo "Could not find ${toolchain_cmake} file, ${_setup_msg}"; exit 1; }

[[ -f ${et_root_dir}/CMakeLists.txt ]] \
    || { echo "Executorch repo doesn't contain CMakeLists.txt file at root level"; exit 1; }

# build executorch libraries
build_executorch
build_quantization_aot_lib

# the test models run, and whether to delegate
test_model=( "softmax" "add" "add3" "mv2" )
test_delegate=( "" "--delegate" "--delegate" "--delegate --quantize" )

# loop over running the AoT flow and executing the model on device
for i in "${!test_model[@]}"; do
    printf "Running e2e flow for model '%s' with flags '%s'\n" "${test_model[i]}" "${test_delegate[i]}"
    pte=$(generate_pte_file "${test_model[i]}" "${test_delegate[i]}")
    # Rebuild the application as the pte is imported as a header/c array
    build_executorch_runner "${pte}"
    run_fvp arm_executor_runner
done

exit 0
