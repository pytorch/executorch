#!/usr/bin/env bash
# All rights reserved.
#
# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

# Temporary workaround until we have a formal Java package
mkdir -p examples/demo-apps/android/ExecuTorchDemo/app/src/main/java/com/example/executorchdemo/executor
cp extension/android/src/main/java/org/pytorch/executorch/*.java examples/demo-apps/android/ExecuTorchDemo/app/src/main/java/com/example/executorchdemo/executor

# Note: Set up ANDROID_NDK, BUCK, and FLATC_EXECUTABLE path
cmake . -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a \
        -DBUCK2=$BUCK \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_FLATC=OFF \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DFLATC_EXECUTABLE=$FLATC_EXECUTABLE \
        -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -Bcmake-out

cmake --build cmake-out -j9

cp cmake-out/extension/android/libexecutorch_jni.so examples/demo-apps/android/ExecuTorchDemo/app/src/main/jniLibs/arm64-v8a/libexecutorch.so
