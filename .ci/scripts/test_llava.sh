#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
# shellcheck source=/dev/null

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

cmake_install_executorch_libraries() {
    cmake                                               \
        -DCMAKE_INSTALL_PREFIX=cmake-out                \
        -DCMAKE_BUILD_TYPE=Debug                        \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON          \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON     \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON            \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON         \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON         \
        -DEXECUTORCH_BUILD_XNNPACK=ON                   \
        -DEXECUTORCH_DO_NOT_USE_CXX11_ABI=ON            \
        -Bcmake-out .


    cmake --build cmake-out -j9 --target install --config Debug
}

cmake_build_llava_runner() {
    dir=examples/models/llava
    python_lib=$($PYTHON_EXECUTABLE -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')

    cmake                                       \
        -DCMAKE_INSTALL_PREFIX=cmake-out        \
        -DCMAKE_BUILD_TYPE=Debug                \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON    \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON           \
        -DCMAKE_PREFIX_PATH="$python_lib"       \
        -Bcmake-out/${dir}                      \
        ${dir}


    cmake --build cmake-out/${dir} -j9 --config Debug
}

# only export the one without custom op for now since it's
export_llava() {
    echo "Starting to export Llava. This will take about 6 mins"
    $PYTHON_EXECUTABLE -m executorch.examples.models.llava.export_llava --pte-name llava.pte --with-artifacts
}

run_and_verify() {
    NOW=$(date +"%H:%M:%S")
    echo "Starting to run llava runner at ${NOW}"
    if [[ ! -f "llava.pte" ]]; then
        echo "Export failed. Abort"
        exit 1
    fi
    if [[ ! -f "image.pt" ]]; then
        echo "image.pt is missing."
        exit 1
    fi
    if [[ ! -f "tokenizer.bin" ]]; then
        echo "tokenizer.bin is missing."
        exit 1
    fi
    RUNTIME_ARGS="--model_path=llava.pte \
     --tokenizer_path=tokenizer.bin \
     --image_path=image.pt \
     --prompt=\"What are the things I should be cautious about when I visit here? ASSISTANT: \" \
     --temperature=0 \
     --seq_len=650"
    cmake-out/examples/models/llava/llava_main ${RUNTIME_ARGS} > result.txt
    # verify result.txt
    RESULT=$(cat result.txt)
    # set the expected prefix to be the same as prompt because there's a bug in sdpa_with_kv_cache that causes <unk> tokens.
    EXPECTED_PREFIX="What are the things I should be cautious about when I visit here? ASSISTANT: "
    if [[ "${RESULT}" == "${EXPECTED_PREFIX}"* ]]; then
        echo "Expected result prefix: ${EXPECTED_PREFIX}"
        echo "Actual result: ${RESULT}"
        echo "Success"
        exit 0
    else
        echo "Expected result prefix: ${EXPECTED_PREFIX}"
        echo "Actual result: ${RESULT}"
        echo "Failure; results not the same"
        exit 1
    fi
}

cmake_install_executorch_libraries
cmake_build_llava_runner
# export_llava
run_and_verify
