#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

CMAKE_OUT="${CMAKE_OUT:-cmake-out-android}"
# Note: Set up ANDROID_NDK and ANDROID_ABI
cmake . -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_CUSTOM=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -B"${CMAKE_OUT}"

if [ "$(uname)" == "Darwin" ]; then
  CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
  CMAKE_JOBS=$(( $(nproc) - 1 ))
fi
cmake --build "${CMAKE_OUT}" -j "${CMAKE_JOBS}" --target install --config Release

cmake examples/models/llama2 \
         -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI="$ANDROID_ABI" \
         -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
         -DCMAKE_BUILD_TYPE=Release \
         -B"${CMAKE_OUT}"/examples/models/llama2

cmake --build "${CMAKE_OUT}"/examples/models/llama2 -j "${CMAKE_JOBS}" --config Release

cmake extension/android \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
  -DEXECUTORCH_BUILD_LLAMA_JNI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -B"${CMAKE_OUT}"/extension/android

cmake --build "${CMAKE_OUT}"/extension/android -j "${CMAKE_JOBS}" --config Release

JNI_LIBS_PATH="examples/demo-apps/android/LlamaDemo/app/src/main/jniLibs"
mkdir -p "${JNI_LIBS_PATH}/${ANDROID_ABI}"
cp "${CMAKE_OUT}"/extension/android/libexecutorch_llama_jni.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
