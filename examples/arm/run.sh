#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu



########
### Hardcoded constants
########
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# Default Ethos-u tool folder override with --scratch-dir=<FOLDER>
root_dir=${script_dir}/ethos-u-scratch

model_name=""
aot_arm_compiler_flags="--delegate --quantize"
portable_kernels="aten::_softmax.out"
target="ethos-u55-128"
output_folder_set=false
output_folder="."
build_with_etdump=false
build_type="Release"
extra_build_flags=""
build_only=false
reorder_inputs=""
system_config=""
memory_mode=""

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --model_name=<MODEL>                   Model to run, can be a builtin, examples/models or a filename Default to all builtin models"
    echo "  --aot_arm_compiler_flags=<FLAGS>       Only used if --model_name is used Default: ${aot_arm_compiler_flags}"
    echo "  --portable_kernels=<OPS>               Comma separated list of portable (non delagated) kernels to include Default: ${portable_kernels}"
    echo "  --target=<TARGET>                      Target to build and run for Default: ${target}"
    echo "  --output=<FOLDER>                      Output folder Default: ${output_folder}"
    echo "  --etdump                               Adds Devtools etdump support to track timing, etdump area will be base64 encoded in the log"
    echo "  --debug_build                          Build with debug flag, default is Release"
    echo "  --extra_build_flags                    Extra flags to pass to cmake like -DET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=60000 Default: none "
    echo "  --build_only                           Only build, don't run FVP"
    echo "  --scratch-dir=<FOLDER>                 Path to your Ethos-U scrach dir if you not using default"
    echo "  --reorder_inputs=<FLAGS>               Reorder the inputs. This can be required when inputs > 1."
    echo "  --system_config=<CONFIG>               System configuration to select from the Vela configuration file (see vela.ini). Default: Ethos_U55_High_End_Embedded for EthosU55 targets, Ethos_U85_SYS_DRAM_Mid for EthosU85 targets."
    echo "                                            NOTE: If given, this option must match the given target. This option also sets timing adapter values customized for specific hardware, see ./executor_runner/CMakeLists.txt."
    echo "  --memory_mode=<MODE>                   Memory mode to select from the Vela configuration file (see vela.ini), e.g. Shared_Sram/Sram_Only. Default: 'Shared_Sram' for Ethos-U55 targets, 'Sram_Only' for Ethos-U85 targets"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --model_name=*) model_name="${arg#*=}";;
      --aot_arm_compiler_flags=*) aot_arm_compiler_flags="${arg#*=}";;
      --portable_kernels=*) portable_kernels="${arg#*=}";;
      --target=*) target="${arg#*=}";;
      --output=*) output_folder="${arg#*=}" ; output_folder_set=true ;;
      --etdump) build_with_etdump=true ;;
      --debug_build) build_type="Debug" ;;
      --extra_build_flags=*) extra_build_flags="${arg#*=}";;
      --build_only) build_only=true ;;
      --scratch-dir=*) root_dir="${arg#*=}";;
      --reorder_inputs=*) reorder_inputs="${arg#*=}";;
      --system_config=*) system_config="${arg#*=}";;
      --memory_mode=*) memory_mode="${arg#*=}";;
      *)
      ;;
    esac
done

root_dir=$(realpath ${root_dir})
output_folder=$(realpath ${output_folder})
mkdir -p ${output_folder}
if [ "$output_folder_set" = true ] ; then
    executor_runner_path=${output_folder}
else
    executor_runner_path=${script_dir}/executor_runner
fi
executor_runner_path=$(realpath ${executor_runner_path})

ethos_u_root_dir="$(cd ${root_dir}/ethos-u && pwd)"
ethos_u_build_dir=${ethos_u_root_dir}/core_platform/build
setup_path_script=${root_dir}/setup_path.sh

# Executorch
et_root_dir=$(cd ${script_dir}/../.. && pwd)
et_build_dir=${et_root_dir}/cmake-out

# Set target based variables
fvp_model=FVP_Corstone_SSE-300_Ethos-U55
if [[ ${target} =~ "ethos-u85" ]]
then
    echo "target is ethos-u85 variant so switching to CS320 FVP"
    fvp_model=FVP_Corstone_SSE-320
fi

if [[ ${system_config} == "" ]]
then
    system_config="Ethos_U55_High_End_Embedded"
    if [[ ${target} =~ "ethos-u85" ]]
    then
        system_config="Ethos_U85_SYS_DRAM_Mid"
    fi
fi

if [[ ${memory_mode} == "" ]]
then
    memory_mode="Shared_Sram"
    if [[ ${target} =~ "ethos-u85" ]]
    then
        system_config="Sram_Only"
    fi
fi

toolchain_cmake=${script_dir}/ethos-u-setup/arm-none-eabi-gcc.cmake
_setup_msg="please refer to ${script_dir}/ethos-u-setup/setup.sh to properly install necessary tools."

if ! [[ $portable_kernels =~ ^((^|,)aten::[a-zA-Z0-9_]+\.[a-zA-Z0-9_]*out)*$ ]]; then
    echo " ERROR: specified argument --portable_kernels=${portable_kernels}"
    echo "        is in the wrong format please use \"aten::<OP1>.out,aten::<OP2>.out,...\""
    echo "        e.g. \"aten::_softmax.out,aten::add.out\""
    exit 1
fi

# Generate a pte file
# output from this function is the pte filename e.g. echo should be avoided or directed to stderr e.g. >&2
function generate_pte_file() {
    [[ $# -ne 2 ]] && { echo "[${FUNCNAME[0]}]" "Expecting model and model_compiler_flags flag, got, $*"; exit 1; }
    local model=${1}
    local model_short_name=$(basename -- "${model}" ".py")
    local model_compiler_flags=${2}

    local model_filename=${model_short_name}_arm_${target}.pte
    if [[ "${model_compiler_flags}" == *"--delegate"* ]]; then
        # Name aligned with default aot_arm_compiler output
        model_filename=${model_short_name}_arm_delegate_${target}.pte
    fi
    cd $et_root_dir

    local pte_file
    pte_file=$(realpath ${output_folder}/${model_filename})
    rm -f "${pte_file}"

    SO_EXT=$(python3 -c 'import platform; print({"Darwin": "dylib", "Linux": "so", "Windows": "dll"}.get(platform.system(), None))')
    # We are using the aot_lib from build_quantization_aot_lib below
    SO_LIB=$(find cmake-out-aot-lib -name libquantized_ops_aot_lib.${SO_EXT})

    local ARM_AOT_CMD="python3 -m examples.arm.aot_arm_compiler --model_name=${model} --target=${target} ${model_compiler_flags} --reorder_inputs=${reorder_inputs} --output ${output_folder} --so_library=$SO_LIB --system_config=${system_config} --memory_mode=${memory_mode}"
    echo "CALL ${ARM_AOT_CMD}" >&2
    ${ARM_AOT_CMD} 1>&2

    [[ -f ${pte_file} ]] || { >&2 echo "Failed to generate a pte file - ${pte_file}"; exit 1; }
    echo "${pte_file}"
}

# build ExecuTorch Libraries
function build_executorch() {
    set -x

    [[ -d "${et_build_dir}" ]] \
        && echo "[${FUNCNAME[0]}] Warn: using already existing build-dir for executorch: ${et_build_dir}!!"
    mkdir -p "${et_build_dir}"

    cd "${et_root_dir}"

    build_with_etdump_flags=""
    if [ "$build_with_etdump" = true ] ; then
        ( set +x ;
            echo "--------------------------------------------------------------------------------" ;
            echo "Build ExecuTorch Libraries host flatcc bin ${build_type} into ${et_root_dir} - cmake-out-host-tools/bin/flatcc" ;
            echo "--------------------------------------------------------------------------------" )


        # Build host flatcc bin
        mkdir -p cmake-out-host-tools
        cmake                                                 \
            -DCMAKE_INSTALL_PREFIX=${et_build_dir}            \
            -DCMAKE_BUILD_TYPE=${build_type}                  \
            -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF            \
            -DEXECUTORCH_ENABLE_LOGGING=ON                    \
            -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON               \
            -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON           \
            -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON       \
            -DEXECUTORCH_BUILD_DEVTOOLS=ON                    \
            -DEXECUTORCH_ENABLE_EVENT_TRACER=ON               \
            -DEXECUTORCH_SEPARATE_FLATCC_HOST_PROJECT=ON      \
            -DFLATCC_ALLOW_WERROR=OFF                         \
            -DFLATC_EXECUTABLE="$(which flatc)"               \
            ${extra_build_flags}                              \
            -Bcmake-out-host-tools                            \
            "${et_root_dir}"

        mkdir -p cmake-out-host-tools/bin
        cp third-party/flatcc/bin/flatcc cmake-out-host-tools/bin

        build_with_etdump_flags="-DEXECUTORCH_BUILD_DEVTOOLS=ON                    \
                                 -DEXECUTORCH_ENABLE_EVENT_TRACER=ON               \
                                 -DEXECUTORCH_SEPARATE_FLATCC_HOST_PROJECT=OFF     \
                                 -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=OFF      \
                                 -DFLATCC_ALLOW_WERROR=OFF                         \
                                 -DFLATCC_EXECUTABLE=${et_root_dir}/cmake-out-host-tools/bin/flatcc "
    fi

    ( set +x ;
        echo "--------------------------------------------------------------------------------" ;
        echo "Build ExecuTorch Libraries target libs with --target install ${build_type} into '${et_root_dir}' - '${et_build_dir}'" ;
        echo "--------------------------------------------------------------------------------" )

    # Build
    cmake                                                 \
        -DCMAKE_INSTALL_PREFIX=${et_build_dir}            \
        -DCMAKE_BUILD_TYPE=${build_type}                  \
        -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"       \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF            \
        -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON               \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON           \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON       \
        -DEXECUTORCH_ENABLE_LOGGING=ON                    \
        ${build_with_etdump_flags}                        \
        -DFLATC_EXECUTABLE="$(which flatc)"               \
        ${extra_build_flags}                              \
        -B${et_build_dir}                                 \
        "${et_root_dir}"

    echo "[${FUNCNAME[0]}] Configured CMAKE"

    cmake --build ${et_build_dir} --parallel --target install --config ${build_type} --

    ( set +x ;
        echo "--------------------------------------------------------------------------------" ;
        echo "Build ExecuTorch Libraries ${build_type} into '${et_root_dir}/examples/arm' - '${et_build_dir}/examples/arm'" ;
        echo "--------------------------------------------------------------------------------" )

    cmake                                                 \
        -DCMAKE_INSTALL_PREFIX=${et_build_dir}            \
        -DCMAKE_BUILD_TYPE=${build_type}                  \
        -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"       \
        -DEXECUTORCH_SELECT_OPS_LIST=${portable_kernels}  \
        -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON               \
        ${extra_build_flags}                              \
        -B"${et_build_dir}/examples/arm"                  \
        "${et_root_dir}/examples/arm"

    cmake --build "${et_build_dir}/examples/arm" --parallel --config ${build_type} --

    set +x

    cd "${et_build_dir}"
    echo "[${FUNCNAME[0]}] Generated static libraries for ExecuTorch:"
    find . -name "*.a" -exec ls -al {} \;
}

# Build .so library to register quant ops with AoT flow
function build_quantization_aot_lib()
{
    SITE_PACKAGES="$(python3 -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
    CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch"

    cd $et_root_dir
    mkdir -p cmake-out-aot-lib

    echo "--------------------------------------------------------------------------------"
    echo "Build .so library to register quant ops with AoT flow ${build_type} into '${et_root_dir}' - 'cmake-out-aot-lib'"
    echo "--------------------------------------------------------------------------------"

    build_with_etdump_flags=""
    if [ "$build_with_etdump" = true ] ; then
        build_with_etdump_flags="-DEXECUTORCH_BUILD_DEVTOOLS=ON        \
                                 -DEXECUTORCH_ENABLE_EVENT_TRACER=ON "
    fi

    cmake \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"    \
        -DCMAKE_BUILD_TYPE=${build_type}            \
        -DEXECUTORCH_BUILD_XNNPACK=OFF              \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON     \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON \
        ${build_with_etdump_flags}                  \
        -DPYTHON_EXECUTABLE=$(which python3)        \
        ${extra_build_flags}                        \
        -Bcmake-out-aot-lib                         \
        "${et_root_dir}"

    cmake --build cmake-out-aot-lib --parallel -- quantized_ops_aot_lib
}

# build Arm Baremetal executor_runner
function build_executorch_runner() {
    echo "[${FUNCNAME[0]}] Generating ExecuTorch libraries"
    [[ $# -ne 1 ]] && { echo "[${FUNCNAME[0]}]" "Expecting a single pte file as argument got, $*"; exit 1; }
    local pte=${1}
    if [[ ${target} == *"ethos-u55"*  ]]; then
        local target_cpu=cortex-m55
    else
        local target_cpu=cortex-m85
    fi
    echo "--------------------------------------------------------------------------------"
    echo "Build Arm Baremetal executor_runner for ${target} - '${executor_runner_path}/cmake-out'"
    echo "--------------------------------------------------------------------------------"

    cd ${script_dir}/executor_runner

    build_with_etdump_flags=""
    if [ "$build_with_etdump" = true ] ; then
        build_with_etdump_flags=" -DEXECUTORCH_ENABLE_EVENT_TRACER=ON "
    fi

    cmake \
      -DCMAKE_BUILD_TYPE=${build_type}            \
      -DCMAKE_TOOLCHAIN_FILE=${toolchain_cmake}   \
      -DTARGET_CPU=${target_cpu}                  \
      -DET_DIR_PATH:PATH=${et_root_dir}           \
      -DET_BUILD_DIR_PATH:PATH=${et_build_dir}    \
      -DET_PTE_FILE_PATH:PATH="${pte}"            \
      -DETHOS_SDK_PATH:PATH=${ethos_u_root_dir}   \
      -DETHOSU_TARGET_NPU_CONFIG=${target}        \
      ${build_with_etdump_flags}                  \
      -DPYTHON_EXECUTABLE=$(which python3)        \
      -DSYSTEM_CONFIG=${system_config}            \
      ${extra_build_flags}                        \
      -B ${executor_runner_path}/cmake-out

    echo "[${FUNCNAME[0]}] Configured CMAKE"

    cmake --build ${executor_runner_path}/cmake-out --parallel -- arm_executor_runner
    echo "[${FUNCNAME[0]}] Generated baremetal elf file:"
    find ${executor_runner_path}/cmake-out -name "arm_executor_runner"
    echo "executable_text: $(find ${executor_runner_path}/cmake-out -name arm_executor_runner -exec arm-none-eabi-size {} \; | grep -v filename | awk '{print $1}') bytes"
    echo "executable_data: $(find ${executor_runner_path}/cmake-out -name arm_executor_runner -exec arm-none-eabi-size {} \; | grep -v filename | awk '{print $2}') bytes"
    echo "executable_bss:  $(find ${executor_runner_path}/cmake-out -name arm_executor_runner -exec arm-none-eabi-size {} \; | grep -v filename | awk '{print $3}') bytes"
}

# Execute the executor_runner on FVP Simulator
function run_fvp() {
    [[ $# -ne 1 ]] && { echo "[${FUNCNAME[0]}]" "Expexted elf binary name, got $*"; exit 1; }
    local elf_name=${1}
    elf=$(find ${executor_runner_path} -name "${elf_name}")
    [[ ! -f $elf ]] && { echo "[${FUNCNAME[0]}]: Unable to find executor_runner elf: ${elf}"; exit 1; }
    num_macs=$(echo ${target} | cut -d - -f 3)

    if [[ ${target} == *"ethos-u55"*  ]]; then
        echo "Running ${elf} for ${target} run with FVP:${fvp_model} num_macs:${num_macs}"
        ${fvp_model}                                            \
            -C ethosu.num_macs=${num_macs}                      \
            -C mps3_board.visualisation.disable-visualisation=1 \
            -C mps3_board.telnetterminal0.start_telnet=0        \
            -C mps3_board.uart0.out_file='-'                    \
            -C mps3_board.uart0.shutdown_on_eot=1               \
            -a "${elf}"                                         \
            --timelimit 220 || true # seconds
        echo "[${FUNCNAME[0]}] Simulation complete, $?"
    elif [[ ${target} == *"ethos-u85"*  ]]; then
        echo "Running ${elf} for ${target} run with FVP:${fvp_model} num_macs:${num_macs}"
    	${fvp_model}                                            \
            -C mps4_board.subsystem.ethosu.num_macs=${num_macs} \
            -C mps4_board.visualisation.disable-visualisation=1 \
            -C vis_hdlcd.disable_visualisation=1                \
            -C mps4_board.telnetterminal0.start_telnet=0        \
            -C mps4_board.uart0.out_file='-'                    \
            -C mps4_board.uart0.shutdown_on_eot=1               \
            -a "${elf}"                                         \
            --timelimit 220 || true # seconds
        echo "[${FUNCNAME[0]}] Simulation complete, $?"
    else
        echo "Running ${elf} for ${target} is not supported"
        exit 1
    fi
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

if [[ -z "$model_name" ]]; then
    # the test models run, and whether to delegate
    test_model=( "softmax" "add" "add3" "mv2" )
    model_compiler_flags=( "" "--delegate" "--delegate" "--delegate --quantize" )
else
    test_model=( "$model_name" )
    model_compiler_flags=( "$aot_arm_compiler_flags" )
    reorder_inputs=( "$reorder_inputs" )
fi

# loop over running the AoT flow and executing the model on device
for i in "${!test_model[@]}"; do
    echo "--------------------------------------------------------------------------------"
    printf "Running e2e flow for model '%s' with flags '%s'\n" "${test_model[i]}" "${model_compiler_flags[i]}"
    echo "--------------------------------------------------------------------------------"
    pte=$(generate_pte_file "${test_model[i]}" "${model_compiler_flags[i]}")
    stat --printf="Generated pte_data_size: %s bytes\npte_file:%n\n" ${pte}
    if [[ ${target} == *"TOSA"*  ]]; then
        echo "Build for ${target} skip generating .elf and running"
    else
        # Rebuild the application as the pte is imported as a header/c array
        build_executorch_runner "${pte}"
        if [ "$build_only" = false ] ; then
            run_fvp arm_executor_runner
        fi
    fi
done

exit 0
