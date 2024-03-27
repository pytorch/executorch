#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

pushd cmake-out
# Note: Set up ANDROID_NDK, ANDROID_ABI, BUCK2, and FLATC_EXECUTABLE
cmake .. -DBUCK2="$BUCK" \
         -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI="$ANDROID_ABI" \
         -DCMAKE_INSTALL_PREFIX=cmake-out \
         -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
         -DEXECUTORCH_BUILD_FLATC=OFF \
         -DFLATC_EXECUTABLE="${FLATC}" \
         -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
         -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
         -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
         -DEXECUTORCH_BUILD_XNNPACK=ON

if [ "$(uname)" == "Darwin" ]; then
  CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
  CMAKE_JOBS=$(( $(nproc) - 1 ))
fi
cmake --build . -j "${CMAKE_JOBS}"
popd

JNI_LIBS_PATH="examples/demo-apps/android/LlamaDemo/app/src/main/jniLibs"
mkdir -p "${JNI_LIBS_PATH}/${ANDROID_ABI}"
cp cmake-out/extension/android/libexecutorch_llama_jni.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"

pushd extension/android
./gradlew build
popd
