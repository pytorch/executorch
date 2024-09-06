#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
# shellcheck source=/dev/null

BUILD_TYPE=${1:-Debug}

echo "Building with BUILD_TYPE: $BUILD_TYPE"

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

cmake_install_executorch_libraries() {
    cmake                                               \
        -DCMAKE_INSTALL_PREFIX=cmake-out                \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}                \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON     \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON          \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON          \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON            \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON         \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON         \
        -DEXECUTORCH_BUILD_XNNPACK=ON                   \
        -DEXECUTORCH_DO_NOT_USE_CXX11_ABI=ON            \
        -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON        \
        -Bcmake-out .


    cmake --build cmake-out -j9 --target install --config ${BUILD_TYPE}
}

cmake_build_llava_runner() {
    dir=examples/models/llava
    python_lib=$($PYTHON_EXECUTABLE -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')

    cmake                                       \
        -DCMAKE_INSTALL_PREFIX=cmake-out        \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}         \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON    \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON           \
        -DCMAKE_PREFIX_PATH="$python_lib"       \
        -Bcmake-out/${dir}                      \
        ${dir}


    cmake --build cmake-out/${dir} -j9 --config ${BUILD_TYPE}
}

# only export the one without custom op for now since it's
export_llava() {
    echo "Starting to export Llava. This will take about 6 mins"
    $PYTHON_EXECUTABLE -m executorch.examples.models.llava.export_llava --pte-name llava.pte --with-artifacts
}

# Download a new image with different size, to test if the model can handle different image sizes
prepare_image_tensor() {
    echo "Downloading image"
    curl -o basketball.jpg https://upload.wikimedia.org/wikipedia/commons/7/73/Chicago_Bulls_and_New_Jersey_Nets%2C_March_28%2C_1991.jpg
    $PYTHON_EXECUTABLE -m executorch.examples.models.llava.image_util --image-path basketball.jpg --output-path image.pt
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
     --prompt=ASSISTANT: \
     --temperature=0 \
     --seq_len=650"
    cmake-out/examples/models/llava/llava_main ${RUNTIME_ARGS} > result.txt
    # verify result.txt
    RESULT=$(cat result.txt)
    # set the expected prefix to be the same as prompt because there's a bug in sdpa_with_kv_cache that causes <unk> tokens.
    if [[ "$(uname)" == "Darwin" ]]; then
        EXPECTED_PREFIX="ASSISTANT: image captures a basketball game in progress, with several players on the court. One of the players is dribbling the ball, while the others are in various"
    else
        # set the expected prefix to be the same as prompt because there's a bug in sdpa_with_kv_cache that causes <unk> tokens.
        EXPECTED_PREFIX="ASSISTANT:"
    fi
    if [[ "${RESULT}" == *"${EXPECTED_PREFIX}"* ]]; then
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
export_llava
prepare_image_tensor
run_and_verify
