#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

BUILD_AAR_DIR="$(mktemp -d)"
export BUILD_AAR_DIR

BASEDIR=$(dirname "$0")
source "$BASEDIR"/../../build/build_android_llm_demo.sh

build_native_library() {
  ANDROID_ABI="$1"
  CMAKE_OUT="cmake-out-android-${ANDROID_ABI}"
  ANDROID_NDK="${ANDROID_NDK:-/opt/ndk}"
  EXECUTORCH_CMAKE_BUILD_TYPE="${EXECUTORCH_CMAKE_BUILD_TYPE:-Release}"
  cmake . -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -B"${CMAKE_OUT}"

  cmake --build "${CMAKE_OUT}" -j16 --target install

  cmake extension/android \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}"/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DCMAKE_INSTALL_PREFIX=c"${CMAKE_OUT}" \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_LLAMA_JNI=ON \
    -B"${CMAKE_OUT}"/extension/android

  cmake --build "${CMAKE_OUT}"/extension/android -j16

  # Copy artifacts to ABI specific directory
  mkdir -p "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}"
  cp "${CMAKE_OUT}"/extension/android/*.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
}

pushd "$BASEDIR"/../../
build_jar
build_native_library "arm64-v8a"
build_native_library "x86_64"
build_aar
bash examples/models/llama/install_requirements.sh
source ".ci/scripts/test_llama.sh" -model stories110M -build_tool cmake -dtype fp16 -mode portable -upload ${BUILD_AAR_DIR}
popd
mkdir -p "$BASEDIR"/src/libs
cp "$BUILD_AAR_DIR/executorch.aar" "$BASEDIR"/src/libs/executorch.aar
python add_model.py
mv "add.pte" "$BASEDIR"/src/androidTest/resources/add.pte
unzip -o "$BUILD_AAR_DIR"/model.zip -d "$BASEDIR"/src/androidTest/resources
