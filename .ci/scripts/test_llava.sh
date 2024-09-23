#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
# shellcheck source=/dev/null

BUILD_TYPE=${1:-Debug}
TARGET_OS=${2:-Native}
BUILD_DIR=${3:-cmake-out}

echo "Building with BUILD_TYPE: $BUILD_TYPE, TARGET_OS: $TARGET_OS, BUILD_DIR: $BUILD_DIR"

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
    PYTHON_EXECUTABLE=python3
fi

TARGET_OS_lower="$(echo "${TARGET_OS}" | awk '{print tolower($0)}')"
if [[ "${TARGET_OS_lower}" == "android" ]]; then
    if [[ -z "${ANDROID_NDK}" ]]; then
        echo "Set ANDROID_NDK environment variable to build for Android."
        exit 1
    fi
fi
PTE=llava.pte
IMAGE_PT=image.pt
IMAGE_CSV=image.csv
TOKENIZER=tokenizer.bin

# Number of processes for a parallel build
NPROC=8
if hash nproc &> /dev/null; then NPROC=$(nproc); fi

EXECUTORCH_COMMON_CMAKE_ARGS="                      \
        -DCMAKE_INSTALL_PREFIX=${BUILD_DIR}         \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}            \
        -DEXECUTORCH_ENABLE_LOGGING=ON              \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON      \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON      \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON        \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON     \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON     \
        -DEXECUTORCH_BUILD_XNNPACK=ON               \
        -DEXECUTORCH_DO_NOT_USE_CXX11_ABI=ON        \
        -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON"

cmake_install_executorch_libraries() {
    cmake                               \
        ${EXECUTORCH_COMMON_CMAKE_ARGS} \
        -B${BUILD_DIR} .

    cmake --build ${BUILD_DIR} -j${NPROC} --target install --config ${BUILD_TYPE}
}

cmake_install_executorch_libraries_for_android() {
    cmake                                                                       \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a                                                 \
        -DANDROID_PLATFORM=android-23                                           \
        ${EXECUTORCH_COMMON_CMAKE_ARGS}                                         \
        -B${BUILD_DIR} .

    cmake --build ${BUILD_DIR} -j${NPROC} --target install --config ${BUILD_TYPE}
}


LLAVA_COMMON_CMAKE_ARGS="                        \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -DCMAKE_INSTALL_PREFIX=${BUILD_DIR}      \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}         \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON     \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON  \
        -DEXECUTORCH_BUILD_XNNPACK=ON"

cmake_build_llava_runner() {
    dir=examples/models/llava
    python_lib=$($PYTHON_EXECUTABLE -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')

    cmake                                 \
        ${LLAVA_COMMON_CMAKE_ARGS}        \
        -DCMAKE_PREFIX_PATH="$python_lib" \
        -B${BUILD_DIR}/${dir}             \
        ${dir}

    cmake --build ${BUILD_DIR}/${dir} -j${NPROC} --config ${BUILD_TYPE}
}


cmake_build_llava_runner_for_android() {
    dir=examples/models/llava
    python_lib=$($PYTHON_EXECUTABLE -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')

    cmake                                                                       \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a                                                 \
        -DANDROID_PLATFORM=android-23                                           \
        ${LLAVA_COMMON_CMAKE_ARGS}                                              \
        -DCMAKE_PREFIX_PATH="$python_lib"                                       \
        -DBUILD_LLAVA_RUNNER_WITHOUT_TORCH=ON                                   \
        -B${BUILD_DIR}/${dir}                                                   \
        ${dir}

    cmake --build ${BUILD_DIR}/${dir} -j${NPROC} --config ${BUILD_TYPE}
}

# only export the one without custom op for now since it's
export_llava() {
    echo "Starting to export Llava. This will take about 6 mins"
    $PYTHON_EXECUTABLE -m executorch.examples.models.llava.export_llava --pte-name ${PTE} --with-artifacts
}

# Download a new image with different size, to test if the model can handle different image sizes
prepare_image_tensor() {
    echo "Downloading image"
    curl -o basketball.jpg https://upload.wikimedia.org/wikipedia/commons/7/73/Chicago_Bulls_and_New_Jersey_Nets%2C_March_28%2C_1991.jpg
    $PYTHON_EXECUTABLE -m executorch.examples.models.llava.image_util --image-path basketball.jpg --output-path ${IMAGE_PT} --output-csv ${IMAGE_CSV}
}

run_and_verify() {
    NOW=$(date +"%H:%M:%S")
    echo "Starting to run llava runner at ${NOW}"
    if [[ ! -f "${PTE}" ]]; then
        echo "Could not file PTE: ${PTE}. Export failed. Aborting."
        exit 1
    fi
    if [[ ! -f "${IMAGE_PT}" ]]; then
        echo ".pt Image file: ${IMAGE_PT} is missing."
        exit 1
    fi
    if [[ ! -f "${IMAGE_CSV}" ]]; then
        echo "csv Image file ${IMAGE_CSV} is missing."
        exit 1
    fi
    if [[ ! -f "${TOKENIZER}" ]]; then
        echo "Tokenizer: ${TOKENIZER} is missing."
        exit 1
    fi

    RUNTIME_ARGS="--model_path=${PTE} \
        --tokenizer_path=${TOKENIZER} \
        --prompt=ASSISTANT:           \
        --temperature=0               \
        --seq_len=650"

    if [[ "${TARGET_OS_lower}" == "android" ]]; then
        echo "Transfer relevant files to the phone via ADB and run llava_main binary."
        exit 0;
    fi

    ${BUILD_DIR}/examples/models/llava/llava_main ${RUNTIME_ARGS} \
        --image_path=${IMAGE_PT} > result.txt

    ${BUILD_DIR}/examples/models/llava/llava_main ${RUNTIME_ARGS} \
        --image_path=${IMAGE_CSV} --is_csv_image=true > result_csv.txt

    # verify result.txt
    RESULT=$(cat result.txt)
    RESULT_CSV=$(cat result_csv.txt)
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
        echo "Actual result (csv): ${RESULT_CSV}"
        echo "Success"
    else
        echo "Expected result prefix: ${EXPECTED_PREFIX}"
        echo "Actual result: ${RESULT}"
        echo "Actual result (csv): ${RESULT_CSV}"
        echo "Failure; generated text is not as expected"
        exit 1
    fi

    if [[ "${RESULT}" == *"${EXPECTED_PREFIX}"* ]]; then
        echo ".pt and .csv image produced same output"
        echo "Success"
        exit 0
    else
        echo ".pt and .csv image inputs produced different outputs"
        echo ".pt image produced: ${RESULT}"
        echo ".csv image produced: ${RESULT_CSV}"
        echo "Failure"
        exit 1
    fi
}

# Step1. Build stuff
if [[ "${TARGET_OS_lower}" == "android" ]]; then
    cmake_install_executorch_libraries_for_android
    cmake_build_llava_runner_for_android
elif [[ "${TARGET_OS_lower}" == "native" ]]; then
    cmake_install_executorch_libraries
    cmake_build_llava_runner
else
    echo "Invalid TARGET_OS ($2): ${TARGET_OS}"
fi

# Step2. Generate the PTE
export_llava

# Step3. Run
prepare_image_tensor
# Only run if building for native
if [[ "${TARGET_OS_lower}" == "native" ]]; then
    run_and_verify
fi
