#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd ${script_dir}/../../.. && pwd)
et_root_dir=$(realpath ${et_root_dir})
setup_path_script=${et_root_dir}/examples/arm/ethos-u-scratch/setup_path.sh
_setup_msg="please refer to ${et_root_dir}/examples/arm/setup.sh to properly install necessary tools."

build_type="Release"
build_with_etdump=false
extra_build_flags=""
output_folder="cmake-out-vkml"

build_with_etdump_flags=" -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF "

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --build_type=<TYPE>             Build with Release, Debug or RelWithDebInfo, default is ${build_type}"
    echo "  --etdump                        Adds Devtools etdump support to track timing, etdump area will be base64 encoded in the log"
    echo "  --extra_build_flags=<FLAGS>     Extra flags to pass to cmake. Default: none "
    echo "  --output=<FOLDER>               Output folder Default: $(output_folder)"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --build_type=*) build_type="${arg#*=}";;
      --etdump) build_with_etdump=true ;;
      --extra_build_flags=*) extra_build_flags="${arg#*=}";;
      --output=*) output_folder="${arg#*=}";;
      --select_ops_list=*) select_ops_list="${arg#*=}";;
      *)
      ;;
    esac
done

# Source the tools
# This should be prepared by the setup.sh
[[ -f ${setup_path_script} ]] \
    || { echo "Missing ${setup_path_script}. ${_setup_msg}"; exit 1; }

source ${setup_path_script}

mkdir -p "${output_folder}"
output_folder=$(realpath ${output_folder})

echo "--------------------------------------------------------------------------------"
echo "Build Arm VKML executor runner: '${output_folder}' with extra build flags: ${extra_build_flags}"
echo "--------------------------------------------------------------------------------"

cd ${et_root_dir}/examples/arm/executor_runner

if [ "$build_with_etdump" = true ] ; then
    build_with_etdump_flags=" -DEXECUTORCH_ENABLE_EVENT_TRACER=ON "
fi

echo "Building with extra flags: ${build_with_etdump_flags} ${extra_build_flags}"
cmake \
    -Wall \
    -Werror \
    -DCMAKE_BUILD_TYPE=${build_type}            \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=OFF \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_VGF=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DPYTHON_EXECUTABLE=$(which python3)        \
    ${extra_build_flags}                        \
    -B ${output_folder} ${et_root_dir}

echo "[${BASH_SOURCE[0]}] Configured CMAKE"

cmake --build ${output_folder} -j$(nproc)

echo "[${BASH_SOURCE[0]}] Built VKML runner: "
find ${output_folder} -name "executor_runner"
