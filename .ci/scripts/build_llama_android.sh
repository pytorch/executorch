#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_executorch_and_backend_lib() {
  echo "Installing executorch and xnnpack backend"
  rm -rf cmake-android-out && mkdir cmake-android-out
  ANDROID_NDK=/opt/ndk
  BUCK2=buck2
  ANDROID_ABI=arm64-v8a
  cmake -DBUCK2="${BUCK2}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -Bcmake-android-out .

  cmake --build cmake-android-out -j16 --target install --config Release
}

build_llama_runner() {
    echo "Building llama runner for Android..."
    ANDROID_ABI=arm64-v8a
    cmake -DBUCK2="$BUCK" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK"/build/cmake/android.toolchain.cmake  \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_OPTIMIZED=ON \
    -Bcmake-android-out/examples/models/llama2 examples/models/llama2

    cmake --build cmake-android-out/examples/models/llama2 -j16 --config Release
}

install_flatc_from_source
install_executorch_and_backend_lib
build_llama_runner
