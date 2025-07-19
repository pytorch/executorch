#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi
which "${PYTHON_EXECUTABLE}"

install_executorch_and_backend_lib() {
  echo "Installing executorch and xnnpack backend"
  clean_executorch_install_folders
  mkdir cmake-android-out
  ANDROID_NDK=${ANDROID_NDK:-/opt/ndk}
  BUCK2=buck2
  ANDROID_ABI=arm64-v8a
  cmake --preset llm \
    -DBUCK2="${BUCK2}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DXNNPACK_ENABLE_ARM_BF16=OFF \
    -Bcmake-android-out .

  cmake --build cmake-android-out -j4 --target install --config Release
}

build_llama_runner() {
    echo "Building llama runner for Android..."
    pushd extension/llm/tokenizers
    echo "Updating tokenizers submodule"
    git submodule update --init
    popd
    ANDROID_ABI=arm64-v8a
    cmake -DBUCK2="${BUCK2}" \
    -DBUILD_TESTING=OFF \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK"/build/cmake/android.toolchain.cmake  \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_BUILD_TYPE=Release \
    -Bcmake-android-out/examples/models/llama examples/models/llama

    cmake --build cmake-android-out/examples/models/llama -j4 --config Release
}
install_executorch_and_backend_lib
build_llama_runner
