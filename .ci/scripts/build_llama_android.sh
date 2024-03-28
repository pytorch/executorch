#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_executorch_and_backend_lib() {
  echo "Installing libexecutorch.a, libportable_kernels.a, libetdump.a, libbundled_program.a"
  rm -rf cmake-android-out && mkdir cmake-android-out
  ANDROID_NDK=/opt/ndk
  BUCK2=$(which buck2)
  FLATC=$(which flatc) ANDROID_ABI=arm64-v8a
  cmake -DBUCK2="$BUCK" -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK"/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_INSTALL_PREFIX=cmake-android-out -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_ENABLE_LOGGING=1 -DEXECUTORCH_BUILD_XNNPACK=ON -DPYTHON_EXECUTABLE=python -DEXECUTORCH_BUILD_OPTIMIZED=ON -DXNNPACK_ENABLE_ARM_BF16=OFF -Bcmake-android-out .

  cmake --build cmake-android-out -j16 --target install --config Release
}

build_llama_runner() {
    echo "Exporting MobilenetV2"
    cmake -DBUCK2=/home/chenlai/bin/buck2 -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK"/build/cmake/android.toolchain.cmake  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_INSTALL_PREFIX=cmake-android-out -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=python -DEXECUTORCH_BUILD_OPTIMIZED=ON -Bcmake-android-out/examples/models/llama2 examples/models/llama2
    cmake --build cmake-android-out/examples/models/llama2 -j16 --config Release
}

install_executorch_and_backend_lib
build_llama_runner
