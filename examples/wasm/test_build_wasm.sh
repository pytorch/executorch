#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

source "$(dirname "${BASH_SOURCE[0]}")/../../.ci/scripts/utils.sh"

test_build_wasm() {
    local model_name=$1
    local model_export_name="${model_name}.pte"
    local model_dir_name="./models_test/"
    echo "Exporting ${model_name}"
    mkdir -p "${model_dir_name}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}" --output_dir="$model_dir_name"

    local example_dir=examples/wasm
    local build_dir=cmake-out/${example_dir}
    rm -rf ${build_dir}
    retry emcmake cmake -DWASM_MODEL_DIR="$(realpath "${model_dir_name}")" -B${build_dir} .

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --target executor_runner

    echo "Removing ${model_dir_name}"
    rm -rf "${model_dir_name}"

    echo 'Running wasm build test'
    $EMSDK_NODE ${build_dir}/executor_runner.js --model_path="${model_export_name}"
}

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

cmake_install_executorch_lib

test_build_wasm add_mul
test_build_wasm mv2
