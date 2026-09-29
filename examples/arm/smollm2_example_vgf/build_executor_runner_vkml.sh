#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Example build helper. This command-line interface is not a public API and may
# change without deprecation.

set -eu

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
et_root_dir=$(cd "${script_dir}/../../.." && pwd)
et_root_dir=$(realpath "${et_root_dir}")

setup_path_script="${et_root_dir}/examples/arm/arm-scratch/setup_path.sh"
output_folder="${1:-cmake-out-vkml}"

[[ -f "${setup_path_script}" ]] \
    || { echo "Missing ${setup_path_script}. Run examples/arm/setup.sh first."; exit 1; }

source "${setup_path_script}"

mkdir -p "${output_folder}"
output_folder=$(realpath "${output_folder}")

cmake \
    -S "${et_root_dir}" \
    -B "${output_folder}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=OFF \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_VGF=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DPYTHON_EXECUTABLE="$(which python3)"

cmake --build "${output_folder}" --parallel

echo "[built] ${output_folder}/executor_runner"
