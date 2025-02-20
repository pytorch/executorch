#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd ${script_dir}/../../.. && pwd)
et_root_dir=$(realpath ${et_root_dir})
toolchain_cmake=${et_root_dir}/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake

pte_file=""
target="ethos-u55-128"
build_type="Release"
system_config=""
build_with_etdump=false
extra_build_flags=""
output_folder_set=false
output_folder="."
et_build_root="${et_root_dir}/arm_test"
ethosu_tools_dir=${et_root_dir}/examples/arm/ethos-u-scratch

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --pte=<PTE_FILE>                pte file (genrated by the aot_arm_compier from the model to include in the elf"
    echo "  --target=<TARGET>               Target to build and run for Default: ${target}"
    echo "  --build_type=<TYPE>             Build with Release, Debug or RelWithDebInfo, default is ${build_type}"
    echo "  --system_config=<CONFIG>        System configuration to select from the Vela configuration file (see vela.ini). Default: Ethos_U55_High_End_Embedded for EthosU55 targets, Ethos_U85_SYS_DRAM_Mid for EthosU85 targets."
    echo "                                     NOTE: If given, this option must match the given target. This option also sets timing adapter values customized for specific hardware, see ./executor_runner/CMakeLists.txt."
    echo "  --etdump                        Adds Devtools etdump support to track timing, etdump area will be base64 encoded in the log"
    echo "  --extra_build_flags=<FLAGS>     Extra flags to pass to cmake like -DET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=60000 Default: none "
    echo "  --output=<FOLDER>               Output folder Default: <MODEL>/<MODEL>_<TARGET INFO>.pte"
    echo "  --et_build_root=<FOLDER>        Build output root folder to use, defaults to ${et_build_root}"
    echo "  --ethosu_tools_dir=<FOLDER>     Path to your Ethos-U tools dir if you not using default: ${ethosu_tools_dir}"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --pte=*) pte_file="${arg#*=}";;
      --target=*) target="${arg#*=}";;
      --build_type=*) build_type="${arg#*=}";;
      --system_config=*) system_config="${arg#*=}";;
      --etdump) build_with_etdump=true ;;
      --extra_build_flags=*) extra_build_flags="${arg#*=}";;
      --output=*) output_folder="${arg#*=}" ; output_folder_set=true ;;
      --et_build_root=*) et_build_root="${arg#*=}";;
      --ethosu_tools_dir=*) ethosu_tools_dir="${arg#*=}";;
      *)
      ;;
    esac
done

pte_file=$(realpath ${pte_file})
ethosu_tools_dir=$(realpath ${ethosu_tools_dir})
ethos_u_root_dir="$ethosu_tools_dir/ethos-u"
ethosu_tools_dir=$(realpath ${ethos_u_root_dir})

et_build_dir=${et_build_root}/cmake-out
et_build_dir=$(realpath ${et_build_dir})

if [ "$output_folder_set" = false ] ; then
    pte_folder=$(cd -- "$( dirname -- "${pte_file}" )" &> /dev/null && pwd)
    pte_short_name=$(basename -- "${pte_file}" ".pte")
    output_folder="$pte_folder/$pte_short_name"
fi

if [[ ${system_config} == "" ]]
then
    system_config="Ethos_U55_High_End_Embedded"
    if [[ ${target} =~ "ethos-u85" ]]
    then
        system_config="Ethos_U85_SYS_DRAM_Mid"
    fi
fi

output_folder=$(realpath ${output_folder})

if [[ ${target} == *"ethos-u55"*  ]]; then
    target_cpu=cortex-m55
else
    target_cpu=cortex-m85
fi
echo "--------------------------------------------------------------------------------"
echo "Build Arm Baremetal executor_runner for ${target} with ${pte_file} using ${system_config} to '${output_folder}/cmake-out'"
echo "--------------------------------------------------------------------------------"

cd ${et_root_dir}/examples/arm/executor_runner

build_with_etdump_flags=""
if [ "$build_with_etdump" = true ] ; then
    echo "Building with etdump e.g. -DEXECUTORCH_ENABLE_EVENT_TRACER=ON"
    build_with_etdump_flags=" -DEXECUTORCH_ENABLE_EVENT_TRACER=ON "
fi

mkdir -p "$output_folder"

cmake \
    -DCMAKE_BUILD_TYPE=${build_type}            \
    -DCMAKE_TOOLCHAIN_FILE=${toolchain_cmake}   \
    -DTARGET_CPU=${target_cpu}                  \
    -DET_DIR_PATH:PATH=${et_root_dir}           \
    -DET_BUILD_DIR_PATH:PATH=${et_build_dir}    \
    -DET_PTE_FILE_PATH:PATH="${pte_file}"            \
    -DETHOS_SDK_PATH:PATH=${ethos_u_root_dir}   \
    -DETHOSU_TARGET_NPU_CONFIG=${target}        \
    ${build_with_etdump_flags}                  \
    -DPYTHON_EXECUTABLE=$(which python3)        \
    -DSYSTEM_CONFIG=${system_config}            \
    ${extra_build_flags}                        \
    -B ${output_folder}/cmake-out

echo "[${BASH_SOURCE[0]}] Configured CMAKE"

cmake --build ${output_folder}/cmake-out --parallel -- arm_executor_runner

echo "[${BASH_SOURCE[0]}] Generated baremetal elf file:"
find ${output_folder}/cmake-out -name "arm_executor_runner"
echo "executable_text: $(find ${output_folder}/cmake-out -name arm_executor_runner -exec arm-none-eabi-size {} \; | grep -v filename | awk '{print $1}') bytes"
echo "executable_data: $(find ${output_folder}/cmake-out -name arm_executor_runner -exec arm-none-eabi-size {} \; | grep -v filename | awk '{print $2}') bytes"
echo "executable_bss:  $(find ${output_folder}/cmake-out -name arm_executor_runner -exec arm-none-eabi-size {} \; | grep -v filename | awk '{print $3}') bytes"
