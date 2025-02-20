#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

########
### Hardcoded constants
########
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd ${script_dir}/../.. && pwd)
et_root_dir=$(realpath ${et_root_dir})


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
system_config=""
memory_mode=""
et_build_root="${et_root_dir}/arm_test"
ethos_u_scratch_dir=${script_dir}/ethos-u-scratch

function help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --model_name=<MODEL>                   Model to run, can be a builtin, examples/models or a filename Default to all builtin models"
    echo "  --aot_arm_compiler_flags=<FLAGS>       Only used if --model_name is used Default: ${aot_arm_compiler_flags}"
    echo "  --portable_kernels=<OPS>               Comma separated list of portable (non delagated) kernels to include Default: ${portable_kernels}"
    echo "  --target=<TARGET>                      Target to build and run for Default: ${target}"
    echo "  --output=<FOLDER>                      Target build output folder Default: ${output_folder}"
    echo "  --etdump                               Adds Devtools etdump support to track timing, etdump area will be base64 encoded in the log"
    echo "  --build_type=<TYPE>                    Build with Release, Debug or RelWithDebInfo, default is ${build_type}"
    echo "  --extra_build_flags=<FLAGS>            Extra flags to pass to cmake like -DET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=60000 Default: none "
    echo "  --build_only                           Only build, don't run FVP"
    echo "  --system_config=<CONFIG>               System configuration to select from the Vela configuration file (see vela.ini). Default: Ethos_U55_High_End_Embedded for EthosU55 targets, Ethos_U85_SYS_DRAM_Mid for EthosU85 targets."
    echo "                                            NOTE: If given, this option must match the given target. This option also sets timing adapter values customized for specific hardware, see ./executor_runner/CMakeLists.txt."
    echo "  --memory_mode=<MODE>                   Memory mode to select from the Vela configuration file (see vela.ini), e.g. Shared_Sram/Sram_Only. Default: 'Shared_Sram' for Ethos-U55 targets, 'Sram_Only' for Ethos-U85 targets"
    echo "  --et_build_root=<FOLDER>               Executorch build output root folder to use, defaults to ${et_build_root}"
    echo "  --scratch-dir=<FOLDER>                 Path to your Ethos-U scrach dir if you not using default ${ethos_u_scratch_dir}"
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
      --build_type=*) build_type="${arg#*=}";;
      --extra_build_flags=*) extra_build_flags="${arg#*=}";;
      --build_only) build_only=true ;;
      --system_config=*) system_config="${arg#*=}";;
      --memory_mode=*) memory_mode="${arg#*=}";;
      --et_build_root=*) et_build_root="${arg#*=}";;
      --scratch-dir=*) ethos_u_scratch_dir="${arg#*=}";;
      *)
      ;;
    esac
done

# Default Ethos-u tool folder override with --scratch-dir=<FOLDER>
ethos_u_scratch_dir=$(realpath ${ethos_u_scratch_dir})
setup_path_script=${ethos_u_scratch_dir}/setup_path.sh
toolchain_cmake=${script_dir}/ethos-u-setup/arm-none-eabi-gcc.cmake
_setup_msg="please refer to ${script_dir}/setup.sh to properly install necessary tools."


# Set target based variables
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
        memory_mode="Sram_Only"
    fi
fi

#######
### Main
#######
# Source the tools
# This should be prepared by the setup.sh
[[ -f ${setup_path_script} ]] \
    || { echo "Missing ${setup_path_script}. ${_setup_msg}"; exit 1; }

source ${setup_path_script}

# basic checks before we get started
hash arm-none-eabi-gcc \
    || { echo "Could not find arm baremetal toolchain on PATH, ${_setup_msg}"; exit 1; }

[[ -f ${toolchain_cmake} ]] \
    || { echo "Could not find ${toolchain_cmake} file, ${_setup_msg}"; exit 1; }

[[ -f ${et_root_dir}/CMakeLists.txt ]] \
    || { echo "Executorch repo doesn't contain CMakeLists.txt file at root level"; exit 1; }

# Build executorch libraries
cd $et_root_dir
if [ "$build_with_etdump" = true ] ; then
    et_dump_flag="--etdump"
else
    et_dump_flag=""
fi

backends/arm/scripts/build_executorch.sh --et_build_root="${et_build_root}" --build_type=$build_type $et_dump_flag
backends/arm/scripts/build_portable_kernels.sh --et_build_root="${et_build_root}" --build_type=$build_type --portable_kernels=$portable_kernels

# Build a lib quantized_ops_aot_lib
backends/arm/scripts/build_quantized_ops_aot_lib.sh --et_build_root="${et_build_root}" --build_type=$build_type

SO_EXT=$(python3 -c 'import platform; print({"Darwin": "dylib", "Linux": "so", "Windows": "dll"}.get(platform.system(), None))')
# We are using the aot_lib from build_quantization_aot_lib below
SO_LIB=$(find "${et_build_root}/cmake-out-aot-lib" -name libquantized_ops_aot_lib.${SO_EXT})


if [[ -z "$model_name" ]]; then
    # the test models run, and whether to delegate
    test_model=( "softmax" "add" "add3" "mv2" )
    model_compiler_flags=( "" "--delegate" "--delegate" "--delegate --quantize" )
else
    test_model=( "$model_name" )
    model_compiler_flags=( "$aot_arm_compiler_flags" )
fi

# loop over running the AoT flow and executing the model on device
for i in "${!test_model[@]}"; do
    model="${test_model[i]}"
    model_compiler_flags="${model_compiler_flags[i]}"

    echo "--------------------------------------------------------------------------------"
    printf "Running e2e flow for model '%s' with flags '%s'\n" "${model}" "${model_compiler_flags}"
    echo "--------------------------------------------------------------------------------"

    cd $et_root_dir
    model_short_name=$(basename -- "${model}" ".py")
    model_filename=${model_short_name}_arm_${target}.pte

    if [[ "${model_compiler_flags}" == *"--delegate"* ]]; then
        # Name aligned with default aot_arm_compiler output
        model_filename=${model_short_name}_arm_delegate_${target}.pte
    fi

    if [ "$output_folder_set" = false ] ; then
        output_folder=${et_build_root}/${model_short_name}
    fi

    output_folder=$(realpath ${output_folder})
    mkdir -p ${output_folder}
    pte_file=$(realpath -m ${output_folder}/${model_filename})

    rm -f "${pte_file}"

    ARM_AOT_CMD="python3 -m examples.arm.aot_arm_compiler --model_name=${model} --target=${target} ${model_compiler_flags} --intermediate=${output_folder} --output=${output_folder} --so_library=$SO_LIB --system_config=${system_config} --memory_mode=${memory_mode}"
    echo "CALL ${ARM_AOT_CMD}" >&2
    ${ARM_AOT_CMD} 1>&2

    [[ -f ${pte_file} ]] || { >&2 echo "Failed to generate a pte file - ${pte_file}"; exit 1; }
    echo "pte_data_size: $(wc -c ${pte_file})"
    echo "pte_file: ${pte_file}"

    if [[ ${target} == *"TOSA"*  ]]; then
        echo "Build for ${target} skip generating a .elf and running it"
    else
        set -x
        # Rebuild the application as the pte is imported as a header/c array
        backends/arm/scripts/build_executorch_runner.sh "--pte=${pte_file}" --build_type=$build_type --target=$target --system_config=$system_config  $et_dump_flag --extra_build_flags="$extra_build_flags" --ethosu_tools_dir="$ethos_u_scratch_dir" --output="${output_folder}"
        if [ "$build_only" = false ] ; then
            # Execute the executor_runner on FVP Simulator
            backends/arm/scripts/run_fvp.sh --elf=${output_folder}/cmake-out/arm_executor_runner --target=$target
        fi
        set +x
    fi
done

exit 0
