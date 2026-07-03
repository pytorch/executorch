#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

########
### Hardcoded constants
########
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd ${script_dir}/../.. && pwd)
et_root_dir=$(realpath "${et_root_dir}")
runner_source_dir="${et_root_dir}/examples/arm/executor_runner/standalone"
runner_source_dir=$(realpath "${runner_source_dir}")

model_name=""
model_input_set=false
model_input=""
aot_arm_compiler_flag_delegate="--delegate"
aot_arm_compiler_flag_quantize="--quantize"
aot_arm_compiler_flags=""
target="ethos-u55-128"
output_folder_set=false
output_folder="."
bundleio=false
build_with_etdump=false
build_type="Release"
build_dir=""
build_only=false
system_config=""
config=""
memory_mode=""
pte_placement="elf"
et_build_root="${et_root_dir}/arm_test"
arm_scratch_dir=${script_dir}/arm-scratch
scratch_dir_set=false
toolchain="arm-none-eabi-gcc"
select_ops_list="aten::_softmax.out"
select_ops_list_overridden=false
qdq_fusion_op=false
model_explorer=false
perf_overlay=false
visualize_tosa=false
visualize_pte=false
model_converter=false
extra_build_flags=""
preset_file="${et_root_dir}/tools/cmake/preset/arm_baremetal.cmake"
cmake_cache_file=""
build_dir_initialized=false
multi_config=false
parallel_jobs=1

function help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --model_name=<MODEL>                   Model file .py/.pth/.pt, builtin model or a model from examples/models. Passed to aot_arm_compiler"
    echo "  --model_input=<INPUT>                  Provide model input .pt file to override the input in the model file. Passed to aot_arm_compiler"
    echo "                                           NOTE: Inference in FVP is done with a dummy input full of ones. Use bundleio flag to run the model in FVP with the custom input or the input from the model file."
    echo "  --aot_arm_compiler_flags=<FLAGS>       Extra flags to pass to aot compiler"
    echo "  --no_delegate                          Do not delegate the model (can't override builtin models)"
    echo "  --no_quantize                          Do not quantize the model (can't override builtin models)"
    echo "  --select_ops_list=<OPS>                Comma separated list of portable (non delegated) kernels to include. Default: ${select_ops_list}"
    echo "                                           NOTE: This is only used when building for semihosting."
    echo "                                           See https://docs.pytorch.org/executorch/stable/kernel-library-selective-build.html for more information."
    echo "  --target=<TARGET>                      Target to build and run for Default: ${target}"
    echo "  --output=<FOLDER>                      Target build output folder Default: ${output_folder}"
    echo "  --bundleio                             Create Bundled pte using Devtools BundelIO with Input/RefOutput included"
    echo "  --etdump                               Adds Devtools etdump support to track timing, etdump area will be base64 encoded in the log"
    echo "  --build_type=<TYPE>                    Build with Release, Debug, RelWithDebInfo, UndefinedSanitizer or AddressSanitizer, default is ${build_type}"
    echo "  --build-dir=<DIR>                      Optional: reuse an existing arm_executor_runner build directory (configured via 'cmake -S examples/arm/executor_runner/standalone -B <DIR> ...'). If omitted, run.sh auto-configures one under ${et_build_root} for bare-metal targets."
    echo "  --build_only                           Only build, don't run"
    echo "  --extra_build_flags=\"<FLAGS>\"         Extra -D style flags to pass to cmake when run.sh auto-configures the build"
    echo "  --toolchain=<arm-none-eabi-gcc|arm-zephyr-eabi-gcc>  Toolchain preset to use when run.sh auto-configures the build. Default: ${toolchain}"
    echo "  --system_config=<CONFIG>               Ethos-U: System configuration to select from the Vela configuration file (see vela.ini). Default: Ethos_U55_High_End_Embedded for EthosU55 targets, Ethos_U65_High_End for EthosU65 targets, Ethos_U85_SYS_DRAM_Mid for EthosU85 targets."
    echo "                                            NOTE: If given, this option must match the given target. This option also sets timing adapter values customized for specific hardware, see ./executor_runner/CMakeLists.txt."
    echo "  --config=<FILEPATH>                    Ethos-U: System configuration file that specifies system configurations (vela.ini)"
    echo "  --memory_mode=<MODE>                   Ethos-U: Memory mode to select from the Vela configuration file (see vela.ini), e.g. Shared_Sram/Sram_Only. Default: 'Shared_Sram' for Ethos-U55 targets, 'Sram_Only' for Ethos-U65 targets and 'Dedicated_Sram_384KB' for Ethos-U85 targets"
    echo "  --pte_placement=<elf|ADDR>             Ethos-U: Control if runtime has PTE baked into the elf or if its placed in memory outside of the elf, defaults to ${pte_placement}"
    echo "  --et_build_root=<FOLDER>               Executorch build output root folder to use, defaults to ${et_build_root}"
    echo "  --scratch-dir=<FOLDER>                 Path to your Ethos-U scratch dir if you not using default ${arm_scratch_dir}"
    echo "  --qdq_fusion_op                        Enable QDQ fusion op"
    echo "  --model_explorer                       Enable model explorer to visualize a TOSA or PTE model graph."
    echo "  --visualize_pte                        With --model_explorer, visualize PTE flatbuffer model and delegates. Cannot be used with --visualize_tosa"
    echo "                                            NOTE: If PTE contains an Ethos-U delegate, the Ethos-U subgraph will be visualized if aot_arm_compiler_flags includes -i for TOSA dumps."
    echo "  --visualize_tosa                       With --model_explorer, visualize TOSA flatbuffer model. Cannot be used with --visualize_pte"
    echo "  --perf_overlay                         With --model_explorer and --visualize_tosa, include performance data from FVP PMU trace."
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
      --select_ops_list=*)
        select_ops_list="${arg#*=}"
        select_ops_list_overridden=true
        ;;
      --target=*) target="${arg#*=}";;
      --output=*) output_folder="${arg#*=}" ; output_folder_set=true ;;
      --bundleio) bundleio=true ;;
      --etdump) build_with_etdump=true ;;
      --build_type=*) build_type="${arg#*=}";;
      --build-dir=*) build_dir="${arg#*=}";;
      --build_only) build_only=true ;;
      --extra_build_flags=*) extra_build_flags="${arg#*=}";;
      --toolchain=*) toolchain="${arg#*=}";;
      --system_config=*) system_config="${arg#*=}";;
      --config=*) config="${arg#*=}";;
      --memory_mode=*) memory_mode="${arg#*=}";;
      --pte_placement=*) pte_placement="${arg#*=}";;
      --et_build_root=*) et_build_root="${arg#*=}";;
      --scratch-dir=*) arm_scratch_dir="${arg#*=}" ; scratch_dir_set=true ;;
      --qdq_fusion_op) qdq_fusion_op=true;;
      --model_explorer) model_explorer=true ;;
      --perf_overlay) perf_overlay=true ;;
      --visualize_tosa) visualize_tosa=true ;;
      --visualize_pte) visualize_pte=true ;;
      *)
      ;;
    esac
done

auto_configure=false
if [[ -z "${build_dir}" ]]; then
    auto_configure=true
fi

if [ "$perf_overlay" = true ] && [ "$model_explorer" != true ]; then
    echo "Error: --perf_overlay requires --model_explorer" >&2
    exit 1
fi

# Cortex-M backend is an operator-library, not a delegate; force-disable
# --delegate when targeting cortex-m so users don't need --no_delegate.
if [[ ${target} == cortex-m* ]]; then
    aot_arm_compiler_flag_delegate=""
fi

if ! [[ ${pte_placement} == "elf" ]]; then
    if ! [[ "$pte_placement" =~ ^0x[0-9a-fA-F]{1,16}$ ]]; then
        echo "ERROR: Placing the PTE in memory failed, address is larger then 64bit $pte_placement"
        exit 1
    fi
    echo "PTE is expected to be placed at $pte_placement instead of the elf."
fi

# Default Ethos-u tool folder override with --scratch-dir=<FOLDER>
arm_scratch_dir=$(realpath "${arm_scratch_dir}")
ethos_u_root_dir="${arm_scratch_dir}/ethos-u"
mkdir -p "${ethos_u_root_dir}"
ethos_u_root_dir=$(realpath "${ethos_u_root_dir}")
cmsis_nn_local_path=""
if [[ -d "${ethos_u_root_dir}/core_software/cmsis-nn" ]]; then
    cmsis_nn_local_path=$(realpath "${ethos_u_root_dir}/core_software/cmsis-nn")
fi
setup_path_script=${arm_scratch_dir}/setup_path.sh
_setup_msg="please refer to ${script_dir}/setup.sh to properly install necessary tools."

toolchain_cmake=""
case "${toolchain}" in
  arm-none-eabi-gcc)
    toolchain_cmake="${et_root_dir}/examples/arm/ethos-u-setup/${toolchain}.cmake"
    ;;
  arm-zephyr-eabi-gcc)
    toolchain_cmake="${et_root_dir}/examples/zephyr/x86_64-linux-arm-zephyr-eabi-gcc.cmake"
    ;;
  *)
    echo "Error: Invalid toolchain selection '${toolchain}'. Valid options: arm-none-eabi-gcc, arm-zephyr-eabi-gcc" >&2
    exit 1
    ;;
esac


# Set target based variables
if [[ ${system_config} == "" ]]
then
    system_config="Ethos_U55_High_End_Embedded"
    if [[ ${target} =~ "ethos-u65" ]]
    then
        system_config="Ethos_U65_High_End"
    fi
    if [[ ${target} =~ "ethos-u85" ]]
    then
        system_config="Ethos_U85_SYS_DRAM_Mid"
    fi
fi

if [[ ${memory_mode} == "" ]]
then
    memory_mode="Shared_Sram"
    if [[ ${target} =~ "ethos-u65" ]]
    then
        memory_mode="Sram_Only"
    fi
    if [[ ${target} =~ "ethos-u85" ]]
    then
        memory_mode="Dedicated_Sram_384KB"
    fi
fi

if [[ ${config} == "" ]]
then
    config="Arm/vela.ini"
fi

target_cpu="cortex-m85"
if [[ ${target} =~ "ethos-u55" || ${target} =~ "ethos-u65" ]]
then
    target_cpu="cortex-m55"
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

    [[ -f ${et_root_dir}/CMakeLists.txt ]] \
        || { echo "Executorch repo doesn't contain CMakeLists.txt file at root level"; return 1; }

    [[ -f ${preset_file} ]] \
        || { echo "Could not find ${preset_file} file, ${_setup_msg}"; return 1; }

    if [[ "${auto_configure}" == true && ${target} != *"TOSA"* ]]; then
        if ! command -v "${toolchain}" >/dev/null 2>&1; then
            echo "Could not find ${toolchain} toolchain on PATH, ${_setup_msg}"
            return 1
        fi

        [[ -f ${toolchain_cmake} ]] \
            || { echo "Could not find ${toolchain_cmake} file, ${_setup_msg}"; return 1; }
    fi

    if [[ ${target} == cortex-m* ]]; then
        # build_test_runner.sh handles toolchain setup; just validate it's on PATH.
        hash arm-none-eabi-gcc \
            || { echo "Could not find arm-none-eabi-gcc on PATH, ${_setup_msg}"; return 1; }
    elif [[ ${target} =~ "vgf" ]]; then
        model_converter=$(which model-converter || true)
        echo "${model_converter}"
        [[ -z "${model_converter}" || "${model_converter}" == "model-converter not found" ]] \
            && { echo "Could not find model-converter, ${_setup_msg}"; return 1; }
    fi

    return 0
}

sanitize_for_path() {
    local value="$1"
    printf '%s' "${value}" | tr -c '[:alnum:]._-' '_'
}

set_default_build_dir_path() {
    if [[ ${target} == *"vgf"* ]]; then
        cat <<EOF >&2
Error: auto-configuring a build directory is only supported for Ethos-U bare-metal targets.
Configure a host build manually, e.g.
  cmake -S "${runner_source_dir}" -B <build-dir> -DEXECUTORCH_ROOT="${et_root_dir}" -DEXECUTORCH_BUILD_VGF=ON
and then pass --build-dir=<build-dir>.
EOF
        exit 1
    fi
    local sanitized_target
    sanitized_target=$(sanitize_for_path "${target}")
    local sanitized_build_type
    sanitized_build_type=$(sanitize_for_path "${build_type}")
    local sanitized_toolchain
    sanitized_toolchain=$(sanitize_for_path "${toolchain}")
    build_dir="${et_build_root}/${sanitized_target}_${sanitized_build_type}_${sanitized_toolchain}"
}

configure_runner_build_dir() {
    local pte_source="$1"
    if [[ -z "${build_dir}" ]]; then
        echo "Error: build_dir is not set. Cannot configure runner." >&2
        exit 1
    fi
    if [[ "${pte_placement}" == "elf" ]]; then
        pte_source=$(realpath "${pte_source}")
    fi
    mkdir -p "${build_dir}"
    local cmake_cmd=(
        cmake -S "${runner_source_dir}" -B "${build_dir}"
        -DEXECUTORCH_ROOT="${et_root_dir}"
        -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"
        -DCMAKE_BUILD_TYPE="${build_type}"
        -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON
        -DEXECUTORCH_BUILD_CORTEX_M=ON
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON
        -DEXECUTORCH_BUILD_PRESET_FILE="${preset_file}"
        -DEXECUTORCH_BAREMETAL_SKIP_INSTALL=OFF
        -DETHOSU_TARGET_NPU_CONFIG="${target}"
        -DTARGET_CPU="${target_cpu}"
        -DSYSTEM_CONFIG="${system_config}"
        -DMEMORY_MODE="${memory_mode}"
        -DETHOS_SDK_PATH:PATH="${ethos_u_root_dir}"
        -DEXECUTORCH_SELECT_OPS_LIST="${select_ops_list}"
    )
    if [[ -n "${cmsis_nn_local_path}" ]]; then
        cmake_cmd+=(-DCMSIS_NN_LOCAL_PATH:PATH="${cmsis_nn_local_path}")
    fi
    cmake_cmd+=(-DET_PTE_FILE_PATH:PATH="${pte_source}")
    if [[ "${pte_placement}" == "elf" ]]; then
        cmake_cmd+=(-DET_MODEL_PTE_ADDR=)
    else
        cmake_cmd+=(-DET_MODEL_PTE_ADDR="${pte_placement}")
    fi
    if [[ "${bundleio}" == true ]]; then
        cmake_cmd+=(-DET_BUNDLE_IO=ON)
    else
        cmake_cmd+=(-DET_BUNDLE_IO=OFF)
    fi
    if [[ "${bundleio}" == true || "${build_with_etdump}" == true ]]; then
        cmake_cmd+=(-DEXECUTORCH_BUILD_DEVTOOLS=ON)
    else
        cmake_cmd+=(-DEXECUTORCH_BUILD_DEVTOOLS=OFF)
    fi
    if [[ "${build_with_etdump}" == true ]]; then
        cmake_cmd+=(-DEXECUTORCH_ENABLE_EVENT_TRACER=ON -DET_DUMP_INTERMEDIATE_OUTPUTS=ON)
    else
        cmake_cmd+=(-DEXECUTORCH_ENABLE_EVENT_TRACER=OFF -DET_DUMP_INTERMEDIATE_OUTPUTS=OFF)
    fi
    if [[ -n "${extra_build_flags}" ]]; then
        # shellcheck disable=SC2206
        local extra_args=(${extra_build_flags})
        cmake_cmd+=("${extra_args[@]}")
    fi
    echo "[run.sh] Configuring ExecuTorch build at ${build_dir}"
    echo "[run.sh] Configuring the build system with ${cmake_cmd[@]}"
    "${cmake_cmd[@]}"
    build_dir_initialized=false
}

cmake_cache_get() {
    local key="$1"
    if [[ ! -f ${cmake_cache_file} ]]; then
        echo ""
        return 0
    fi
    local line
    line=$(grep -m1 "^${key}:" "${cmake_cache_file}" || true)
    if [[ -z "${line}" ]]; then
        echo ""
    else
        echo "${line#*=}"
    fi
}

cmake_cache_has_key() {
    local key="$1"
    [[ -f ${cmake_cache_file} ]] && grep -q "^${key}:" "${cmake_cache_file}"
}

ensure_runner_build_dir() {
    local standalone
    standalone=$(cmake_cache_get ARM_EXECUTOR_RUNNER_STANDALONE)
    local normalized
    normalized=$(printf '%s' "${standalone}" | tr '[:lower:]' '[:upper:]')
    if [[ "${normalized}" != "TRUE" && "${normalized}" != "ON" ]]; then
        cat <<EOF >&2
Error: ${build_dir} is not a standalone arm_executor_runner build directory.
Configure it via:
  cmake -S ${runner_source_dir} -B ${build_dir} -DEXECUTORCH_ROOT=${et_root_dir} [...]
and re-run run.sh.
EOF
        exit 1
    fi
}

ensure_select_ops_list_setting() {
    local expected="$1"
    local cache_value
    cache_value=$(cmake_cache_get EXECUTORCH_SELECT_OPS_LIST)
    if [[ -z "${cache_value}" ]]; then
        cat <<EOF >&2
Error: EXECUTORCH_SELECT_OPS_LIST is not configured in ${build_dir}.
Reconfigure cmake -S ${runner_source_dir} -B ${build_dir} -DEXECUTORCH_SELECT_OPS_LIST=${expected}.
EOF
        exit 1
    fi
    if [[ "${cache_value}" != "${expected}" ]]; then
        cat <<EOF >&2
Error: ${build_dir} was configured with EXECUTORCH_SELECT_OPS_LIST=${cache_value}, but run.sh requested ${expected}.
Reconfigure cmake -S ${runner_source_dir} -B ${build_dir} -DEXECUTORCH_SELECT_OPS_LIST=${expected}, or omit --select_ops_list.
EOF
        exit 1
    fi
}

require_cache_value() {
    local key="$1"
    local expected="$2"
    local value
    if ! cmake_cache_has_key "${key}"; then
        echo "Error: ${key} not found in ${cmake_cache_file}. Reconfigure CMake with -D${key}=${expected}." >&2
        exit 1
    fi
    value=$(cmake_cache_get "${key}")
    if [[ "${value}" != "${expected}" ]]; then
        echo "Error: ${key}=${value} in ${build_dir}. Reconfigure CMake with -D${key}=${expected} to use this run.sh invocation." >&2
        exit 1
    fi
}

require_cache_bool() {
    local key="$1"
    local expected="$2"
    local value
    value=$(cmake_cache_get "${key}")
    if [[ -z "${value}" ]]; then
        echo "Error: ${key} not found in ${cmake_cache_file}. Reconfigure CMake with -D${key}=${expected}." >&2
        exit 1
    fi
    local value_upper
    value_upper=$(printf '%s' "${value}" | tr '[:lower:]' '[:upper:]')
    local expected_upper
    expected_upper=$(printf '%s' "${expected}" | tr '[:lower:]' '[:upper:]')
    if [[ "${value_upper}" != "${expected_upper}" ]]; then
        echo "Error: ${key}=${value} in ${build_dir}. Reconfigure CMake with -D${key}=${expected} to use run.sh." >&2
        exit 1
    fi
}

is_cmake_false_value() {
    local value_upper
    value_upper=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')
    case "${value_upper}" in
        ""|0|OFF|FALSE|NO|N|IGNORE|*-NOTFOUND)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

ensure_pte_placement_setting() {
    local cached_addr
    cached_addr=$(cmake_cache_get ET_MODEL_PTE_ADDR)
    if ! cmake_cache_has_key ET_MODEL_PTE_ADDR; then
        echo "Error: ET_MODEL_PTE_ADDR not found in ${cmake_cache_file}. Reconfigure CMake for the requested --pte_placement=${pte_placement}." >&2
        exit 1
    fi
    if [[ "${pte_placement}" == "elf" ]]; then
        if ! is_cmake_false_value "${cached_addr}"; then
            echo "Error: --pte_placement=elf requested, but ${build_dir} was configured with ET_MODEL_PTE_ADDR=${cached_addr}. Reconfigure CMake with -DET_MODEL_PTE_ADDR=." >&2
            exit 1
        fi
        if ! cmake_cache_has_key ET_PTE_FILE_PATH; then
            echo "Error: ET_PTE_FILE_PATH not found in ${cmake_cache_file}. Reconfigure CMake with -DET_PTE_FILE_PATH=<model.pte>." >&2
            exit 1
        fi
        return
    fi
    if is_cmake_false_value "${cached_addr}"; then
        echo "Error: --pte_placement=${pte_placement} requested, but ${build_dir} was configured for an embedded PTE. Reconfigure CMake with -DET_MODEL_PTE_ADDR=${pte_placement}, or use --pte_placement=elf." >&2
        exit 1
    fi
    if [[ "${cached_addr}" != "${pte_placement}" ]]; then
        echo "Error: --pte_placement=${pte_placement} requested, but ${build_dir} was configured with ET_MODEL_PTE_ADDR=${cached_addr}. Reconfigure CMake with -DET_MODEL_PTE_ADDR=${pte_placement}." >&2
        exit 1
    fi
}

get_parallel_jobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1 && sysctl hw.logicalcpu >/dev/null 2>&1; then
        sysctl -n hw.logicalcpu
    elif command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN
    elif [[ -n "${NUMBER_OF_PROCESSORS:-}" ]]; then
        echo "${NUMBER_OF_PROCESSORS}"
    else
        echo 1
    fi
}

build_runner_target() {
    local cmake_target="$1"
    local build_cmd=(cmake --build "${build_dir}" --target "${cmake_target}" --parallel "${parallel_jobs}")
    if [[ "${multi_config}" == true ]]; then
        build_cmd+=(--config "${build_type}")
    fi
    echo "[run.sh] Building target ${cmake_target} in ${build_dir}"
    "${build_cmd[@]}"
}

locate_runner_binary() {
    local binary_name="$1"
    local candidates=()
    if [[ "${multi_config}" == true ]]; then
        candidates+=("${build_dir}/${build_type}/${binary_name}")
        candidates+=("${build_dir}/examples/arm/executor_runner/${build_type}/${binary_name}")
    fi
    candidates+=("${build_dir}/${binary_name}")
    candidates+=("${build_dir}/examples/arm/executor_runner/${binary_name}")
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    local found
    found=$(find "${build_dir}" -name "${binary_name}" -type f 2>/dev/null | head -n 1 || true)
    if [[ -n "${found}" ]]; then
        echo "${found}"
        return 0
    fi
    return 1
}
ensure_build_dir_ready() {
    if [[ "${build_dir_initialized}" == true ]]; then
        return
    fi
    if [[ -z "${build_dir}" ]]; then
        echo "Error: build_dir is not set. Configure CMake first." >&2
        exit 1
    fi
    build_dir=$(realpath "${build_dir}")
    cmake_cache_file="${build_dir}/CMakeCache.txt"
    if [[ ! -f ${cmake_cache_file} ]]; then
        cat <<EOF >&2
Error: ${build_dir} does not contain a configured arm_executor_runner build (missing CMakeCache.txt).
Run cmake -S ${runner_source_dir} -B ${build_dir} -DEXECUTORCH_ROOT=${et_root_dir} with the desired options first, then re-run run.sh.
EOF
        exit 1
    fi
    if [[ ${target} == *"vgf"* ]]; then
        require_cache_bool EXECUTORCH_BUILD_VGF ON
    else
        ensure_runner_build_dir
        require_cache_bool EXECUTORCH_BUILD_ARM_BAREMETAL ON
        require_cache_bool EXECUTORCH_BAREMETAL_SKIP_INSTALL OFF
        require_cache_value ETHOSU_TARGET_NPU_CONFIG "${target}"
        require_cache_value TARGET_CPU "${target_cpu}"
        require_cache_value SYSTEM_CONFIG "${system_config}"
        require_cache_value MEMORY_MODE "${memory_mode}"
        if [[ "${bundleio}" == true ]]; then
            require_cache_bool ET_BUNDLE_IO ON
        else
            require_cache_bool ET_BUNDLE_IO OFF
        fi
        if [[ "${bundleio}" == true || "${build_with_etdump}" == true ]]; then
            require_cache_bool EXECUTORCH_BUILD_DEVTOOLS ON
        else
            require_cache_bool EXECUTORCH_BUILD_DEVTOOLS OFF
        fi
        if [[ "${build_with_etdump}" == true ]]; then
            require_cache_bool EXECUTORCH_ENABLE_EVENT_TRACER ON
            require_cache_bool ET_DUMP_INTERMEDIATE_OUTPUTS ON
        else
            require_cache_bool EXECUTORCH_ENABLE_EVENT_TRACER OFF
            require_cache_bool ET_DUMP_INTERMEDIATE_OUTPUTS OFF
        fi
    fi
    if [[ ${target} != *"vgf"* ]]; then
        ensure_select_ops_list_setting "${select_ops_list}"
    fi
    multi_config=false
    if [[ -n "$(cmake_cache_get CMAKE_CONFIGURATION_TYPES)" ]]; then
        multi_config=true
    fi
    parallel_jobs=$(get_parallel_jobs)
    build_dir_initialized=true
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

cd "${et_root_dir}"

bundleio_flag=""
etrecord_flag_template=""
qdq_fusion_op_flag=""
if [ "$build_with_etdump" = true ] ; then
    etrecord_flag_template="--etrecord"
fi

if [ "$bundleio" = true ] ; then
    bundleio_flag="--bundleio"
fi

if [ "$qdq_fusion_op" = true ] ; then
    qdq_fusion_op_flag="--enable_qdq_fusion_pass"
fi

if [[ "${auto_configure}" == true ]]; then
    set_default_build_dir_path
else
    if [[ -z "${build_dir}" ]]; then
        echo "Error: --build-dir must not be empty." >&2
        exit 1
    fi
    ensure_build_dir_ready
fi

stage_pte_into_cache() {
    local new_pte="$1"
    local cache_path
    cache_path=$(cmake_cache_get ET_PTE_FILE_PATH)
    if [[ -z "${cache_path}" ]]; then
        cat <<EOF >&2
Error: --pte_placement=elf requires ET_PTE_FILE_PATH to be set when configuring CMake.
Re-run cmake -S . -B ${build_dir} -DET_PTE_FILE_PATH=/absolute/path/to/model.pte (or use --pte_placement=<addr>).
EOF
        exit 1
    fi
    if [[ "${cache_path}" != /* ]]; then
        cache_path="${build_dir}/${cache_path}"
    fi
    mkdir -p "$(dirname "${cache_path}")"
    cp "${new_pte}" "${cache_path}"
    echo "${cache_path}"
}

if [[ -z "$model_name" ]]; then
    echo "[run.sh] WARNING: Built-in test models executed when --model_name is omitted are deprecated and will be removed after the ExecuTorch 1.2 release." >&2
    # the test models run, and whether to delegate
    test_model=(
        "softmax"   # 0
        "add"       # 1
        "qadd"      # 3
        "qadd2"     # 4
        "qops"      # 5
        "mv2"       # 6
    )
    model_compiler_flags=(
        ""                      # 0 softmax
        "--delegate"            # 1 add
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

    cd "${et_root_dir}"
    # Remove path and file exetension to get model_short_name
    ext=${model##*.}
    model_short_name=$(basename -- "${model}" .$ext)
    model_filename=${model_short_name}_arm_${target}

    if [[ "${model_compiler_flags}" == *"--delegate"* ]]; then
        # Name aligned with default aot_arm_compiler output
        model_filename=${model_short_name}_arm_delegate_${target}
    fi

    if [ "$bundleio" = true ] ; then
        model_filename_ext=${model_filename}.bpte
    else
        model_filename_ext=${model_filename}.pte
    fi

    if [ "$output_folder_set" = false ] ; then
        output_folder=${et_build_root}/${model_short_name}
    fi

    local_fvp_pmu_flag=""
    if [ "$perf_overlay" = true ] ; then
        model_compiler_flags+="--enable_debug_mode tosa"
        local_fvp_pmu_flag="--trace_file=${output_folder}/pmu_trace.gz"
    fi

    mkdir -p "${output_folder}"
    output_folder=$(realpath "${output_folder}")
    pte_file="${output_folder}/${model_filename_ext}"

    # Remove old pte files
    rm -f "${output_folder}/${model_filename_ext}"

    if [ "$model_input_set" = true ]; then
        model_compiler_flags="${model_compiler_flags} --model_input=${model_input}"
    fi

    model_etrecord_flag="${etrecord_flag_template}"
    ARM_AOT_CMD="python3 -m backends.arm.scripts.aot_arm_compiler --model_name=${model} --target=${target} ${model_compiler_flags} --intermediate=${output_folder} --output=${pte_file} --system_config=${system_config} --memory_mode=${memory_mode} $bundleio_flag ${model_etrecord_flag} --config=${config} $qdq_fusion_op_flag"
    echo "CALL ${ARM_AOT_CMD}" >&2
    ${ARM_AOT_CMD} 1>&2

    pte_file=$(realpath "${pte_file}")

    if [ "${model_etrecord_flag}" != "" ] ; then
        etrecord_filename="${output_folder}/${model_filename}_etrecord.bin"
        etrecord_filename=$(realpath "${etrecord_filename}")
        model_etrecord_flag="--etrecord=${etrecord_filename}"
    fi

    [[ -f ${pte_file} ]] || { >&2 echo "Failed to generate a pte file - ${pte_file}"; exit 1; }
    echo "pte_data_size: $(wc -c ${pte_file})"
    echo "pte_file: ${pte_file}"

    if [[ ${target} == *"TOSA"*  ]]; then
        echo "Build for ${target} skip generating a .elf and running it"
        continue
    elif [[ ${target} == cortex-m*  ]]; then
        # Cortex-M backend uses a semihosting executor_runner (built by
        # build_test_runner.sh, one per target) that loads the .bpte at
        # runtime, rather than per-model runners with the PTE baked in.
        if [ "$bundleio" != true ]; then
            echo "Error: --target=${target} requires --bundleio (the cortex-m runner loads bundled inputs via semihosting)"
            exit 1
        fi
        set -x
        backends/cortex_m/test/build_test_runner.sh --target="${target}"
        cortex_m_elf="${et_root_dir}/arm_test/arm_semihosting_executor_runner_corstone-300_${target}/arm_executor_runner"
        if [ "$build_only" = false ] ; then
            backends/arm/scripts/run_fvp.sh --elf="${cortex_m_elf}" --target="${target}" --bundle="${pte_file}"
        fi
        set +x
    elif [[ ${target} == *"vgf"*  ]]; then
        echo "Build and run for VKML, (target: ${target})"
        build_runner_target executor_runner
        if [ "$build_only" = false ] ; then
            backends/arm/scripts/run_vkml.sh --model=${pte_file} --build_path=${build_dir}
        fi
    else
        if [[ "${auto_configure}" == true ]]; then
            configure_runner_build_dir "${pte_file}"
        fi
        ensure_build_dir_ready
        ensure_pte_placement_setting

        model_data=""
        if [[ ${pte_placement} == "elf" ]]; then
            if [[ "${auto_configure}" == true ]]; then
                staged_path=$(cmake_cache_get ET_PTE_FILE_PATH)
                echo "ET_PTE_FILE_PATH payload: ${staged_path}"
            else
                staged_path=$(stage_pte_into_cache "${pte_file}")
                echo "Updated ET_PTE_FILE_PATH payload: ${staged_path}"
            fi
        else
            model_data="--data=${pte_file}@${pte_placement}"
        fi


        build_runner_target arm_executor_runner
        elf_file=$(locate_runner_binary arm_executor_runner) \
            || { echo "Failed to locate arm_executor_runner in ${build_dir}." >&2; exit 1; }
        if [ "$build_only" = false ] ; then
            fvp_args=("--elf=${elf_file}" "--target=${target}")
            if [[ -n "${model_data}" ]]; then
                fvp_args+=("${model_data}")
            fi
            if [[ -n "${model_etrecord_flag}" ]]; then
                fvp_args+=("${model_etrecord_flag}")
            fi
            if [[ -n "${local_fvp_pmu_flag}" ]]; then
                fvp_args+=("${local_fvp_pmu_flag}")
            fi
            backends/arm/scripts/run_fvp.sh "${fvp_args[@]}"
        fi
    fi

    if [ "$model_explorer" = true ]; then
        perf_flags=""
        if [ "$perf_overlay" = true ]; then
            perf_flags+=" --trace ${output_folder}/pmu_trace.gz --tables ${output_folder}/output/out_debug.xml"
        fi

        visualization_file=""
        if [ "$visualize_tosa" = true ]; then
            visualization_file+=" --tosa"
        fi
        if [ "$visualize_pte" = true ]; then
            visualization_file+=" --pte"
        fi

        me_flags="${visualization_file} ${perf_flags}"
        python3 ${script_dir}/visualize.py --model_dir ${output_folder} ${me_flags}
    fi
done

exit 0
