#!/usr/bin/env bash
# All rights reserved.
#
# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

EXECUTOR_JAVA_DIR=examples/demo-apps/android/ExecuTorchDemo/app/src/main/java/com/example/executorchdemo/executor/
mkdir -p "${EXECUTOR_JAVA_DIR}"
# Temporary workaround until we have a formal Java package
cp extension/android/src/main/java/org/pytorch/executorch/*.java "${EXECUTOR_JAVA_DIR}"

pushd cmake-out
# Note: Set up ANDROID_NDK, ANDROID_ABI, BUCK2, and FLATC_EXECUTABLE
cmake .. -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DBUCK2="${BUCK2}" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_FLATC=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DFLATC_EXECUTABLE="${FLATC}" \
  -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON

if [ "$(uname)" == "Darwin" ]; then
  CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
  CMAKE_JOBS=$(( $(nproc) - 1 ))
fi
cmake --build . -j "${CMAKE_JOBS}"
popd

JNI_LIBS_PATH="examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs"
mkdir -p "${JNI_LIBS_PATH}/${ANDROID_ABI}"
cp cmake-out/extension/android/libexecutorch_jni.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/libexecutorch.so"
