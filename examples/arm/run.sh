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
model_input_set=false
model_input=""
aot_arm_compiler_flag_delegate="--delegate"
aot_arm_compiler_flag_quantize="--quantize"
aot_arm_compiler_flags=""
portable_kernels="aten::_softmax.out"
target="ethos-u55-128"
output_folder_set=false
output_folder="."
bundleio=false
build_with_etdump=false
build_type="Release"
extra_build_flags=""
build_only=false
system_config=""
memory_mode=""
et_build_root="${et_root_dir}/arm_test"
ethos_u_scratch_dir=${script_dir}/ethos-u-scratch
scratch_dir_set=false

function help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --model_name=<MODEL>                   Model file .py/.pth/.pt, builtin model or a model from examples/models. Passed to aot_arm_compiler"
    echo "  --model_input=<INPUT>                  Provide model input .pt file to override the input in the model file. Passed to aot_arm_compiler"
    echo "                                           NOTE: Inference in FVP is done with a dummy input full of ones. Use bundleio flag to run the model in FVP with the custom input or the input from the model file."
    echo "  --aot_arm_compiler_flags=<FLAGS>       Extra flags to pass to aot compiler"
    echo "  --no_delegate                          Do not delegate the model (can't override builtin models)"
    echo "  --no_quantize                          Do not quantize the model (can't override builtin models)"
    echo "  --portable_kernels=<OPS>               Comma separated list of portable (non delagated) kernels to include Default: ${portable_kernels}"
    echo "  --target=<TARGET>                      Target to build and run for Default: ${target}"
    echo "  --output=<FOLDER>                      Target build output folder Default: ${output_folder}"
    echo "  --bundleio                             Create Bundled pte using Devtools BundelIO with Input/RefOutput included"
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
      --model_input=*) model_input="${arg#*=}" ; model_input_set=true  ;;
      --aot_arm_compiler_flags=*) aot_arm_compiler_flags="${arg#*=}";;
      --no_delegate) aot_arm_compiler_flag_delegate="" ;;
      --no_quantize) aot_arm_compiler_flag_quantize="" ;;
      --portable_kernels=*) portable_kernels="${arg#*=}";;
      --target=*) target="${arg#*=}";;
      --output=*) output_folder="${arg#*=}" ; output_folder_set=true ;;
      --bundleio) bundleio=true ;;
      --etdump) build_with_etdump=true ;;
      --build_type=*) build_type="${arg#*=}";;
      --extra_build_flags=*) extra_build_flags="${arg#*=}";;
      --build_only) build_only=true ;;
      --system_config=*) system_config="${arg#*=}";;
      --memory_mode=*) memory_mode="${arg#*=}";;
      --et_build_root=*) et_build_root="${arg#*=}";;
      --scratch-dir=*) ethos_u_scratch_dir="${arg#*=}" ; scratch_dir_set=true ;;
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
        memory_mode="Dedicated_Sram_384KB"
    fi
fi

function check_setup () {
    # basic checks that setup.sh did everything needed before we get started

    # check if setup_path_script was created, if so source it
    if [[ -f ${setup_path_script} ]]; then
        source $setup_path_script
    else
        echo "Could not find ${setup_path_script} file, ${_setup_msg}"
        return 1
    fi

    # If setup_path_script was correct all these checks should now pass
    hash arm-none-eabi-gcc \
        || { echo "Could not find arm baremetal toolchain on PATH, ${_setup_msg}"; return 1; }

    [[ -f ${toolchain_cmake} ]] \
        || { echo "Could not find ${toolchain_cmake} file, ${_setup_msg}"; return 1; }

    [[ -f ${et_root_dir}/CMakeLists.txt ]] \
        || { echo "Executorch repo doesn't contain CMakeLists.txt file at root level"; return 1; }

    return 0
}

#######
### Main
#######
if ! check_setup; then
    if [ "$scratch_dir_set" = false ] ; then
	# check setup failed, no scratchdir given as parameter. trying to run setup.sh
	if ${script_dir}/setup.sh; then
	    # and recheck setup. If this fails exit.
	    if ! check_setup; then
		exit 1
	    fi
	else
	    # setup.sh failed, it should print why
	    exit 1
	fi
    fi
fi

# Build executorch libraries
cd $et_root_dir
devtools_flag=""
bundleio_flag=""
et_dump_flag=""
if [ "$build_with_etdump" = true ] ; then
    devtools_flag="--devtools --etdump"
    et_dump_flag="--etdump"
fi

if [ "$bundleio" = true ] ; then
    devtools_flag="--devtools --etdump"
    bundleio_flag="--bundleio"
    et_dump_flag="--etdump"
fi

backends/arm/scripts/build_executorch.sh --et_build_root="${et_build_root}" --build_type=$build_type $devtools_flag
backends/arm/scripts/build_portable_kernels.sh --et_build_root="${et_build_root}" --build_type=$build_type --portable_kernels=$portable_kernels

if [[ -z "$model_name" ]]; then
    # the test models run, and whether to delegate
    test_model=(
        "softmax"  # 0
        "add"      # 1
        "add3"     # 2
        "qadd"     # 3
        "qadd2"    # 4
        "qops"     # 5
        "mv2"      # 6
    )
    model_compiler_flags=(
        ""                      # 0 softmax
        "--delegate"            # 1 add
        "--delegate"            # 2 add3
        "--delegate --quantize" # 3 qadd
        "--delegate --quantize" # 4 qadd2
        "--delegate --quantize" # 5 qops
        "--delegate --quantize" # 6 mv2
    )
else
    test_model=( "$model_name" )
    model_compiler_flags=( "$aot_arm_compiler_flag_delegate $aot_arm_compiler_flag_quantize $aot_arm_compiler_flags" )
fi

# loop over running the AoT flow and executing the model on device
for i in "${!test_model[@]}"; do
    model="${test_model[i]}"
    model_compiler_flags="${model_compiler_flags[i]}"

    echo "--------------------------------------------------------------------------------"
    printf "Running e2e flow for model '%s' with flags '%s'\n" "${model}" "${model_compiler_flags}"
    echo "--------------------------------------------------------------------------------"

    cd $et_root_dir
    # Remove path and file exetension to get model_short_name
    ext=${model##*.}
    model_short_name=$(basename -- "${model}" .$ext)
    model_filename=${model_short_name}_arm_${target}

    if [[ "${model_compiler_flags}" == *"--delegate"* ]]; then
        # Name aligned with default aot_arm_compiler output
        model_filename=${model_short_name}_arm_delegate_${target}
    fi
    elf_folder=${model_filename}

    if [ "$bundleio" = true ] ; then
        model_filename=${model_filename}.bpte
    else
        model_filename=${model_filename}.pte
    fi

    if [ "$output_folder_set" = false ] ; then
        output_folder=${et_build_root}/${model_short_name}
    fi

    mkdir -p ${output_folder}
    output_folder=$(realpath ${output_folder})
    pte_file="${output_folder}/${model_filename}"

    # Remove old pte files
    rm -f "${output_folder}/${model_filename}"

    if [ "$model_input_set" = true ]; then
        model_compiler_flags="${model_compiler_flags} --model_input=${model_input}"
    fi

    ARM_AOT_CMD="python3 -m examples.arm.aot_arm_compiler --model_name=${model} --target=${target} ${model_compiler_flags} --intermediate=${output_folder} --output=${pte_file} --system_config=${system_config} --memory_mode=${memory_mode} $bundleio_flag"
    echo "CALL ${ARM_AOT_CMD}" >&2
    ${ARM_AOT_CMD} 1>&2

    pte_file=$(realpath ${pte_file})

    [[ -f ${pte_file} ]] || { >&2 echo "Failed to generate a pte file - ${pte_file}"; exit 1; }
    echo "pte_data_size: $(wc -c ${pte_file})"
    echo "pte_file: ${pte_file}"

    if [[ ${target} == *"TOSA"*  ]]; then
        echo "Build for ${target} skip generating a .elf and running it"
    else
        set -x
        # Rebuild the application as the pte is imported as a header/c array
        backends/arm/scripts/build_executor_runner.sh --et_build_root="${et_build_root}" --pte="${pte_file}" --build_type=${build_type} --target=${target} --system_config=${system_config} --memory_mode=${memory_mode} ${bundleio_flag} ${et_dump_flag} --extra_build_flags="${extra_build_flags}" --ethosu_tools_dir="${ethos_u_scratch_dir}"
        if [ "$build_only" = false ] ; then
            # Execute the executor_runner on FVP Simulator
            elf_file="${output_folder}/${elf_folder}/cmake-out/arm_executor_runner"
            backends/arm/scripts/run_fvp.sh --elf=${elf_file} --target=$target
        fi
        set +x
    fi
done

exit 0
