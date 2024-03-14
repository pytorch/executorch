#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# https://github.com/pytorch/executorch/tree/main/examples/demo-apps/android/ExecuTorchDemo
build_executorch_android_demo_app() {
  MODEL_NAME=dl3
  # Delegating DeepLab v3 to XNNPACK backend
  python -m examples.xnnpack.aot_compiler --model_name="${MODEL_NAME}" --delegate

  ASSETS_DIR=examples/demo-apps/android/ExecuTorchDemo/app/src/main/assets/
  mkdir -p "${ASSETS_DIR}"
  cp "${MODEL_NAME}_xnnpack_fp32.pte" "${ASSETS_DIR}"

  EXECUTOR_JAVA_DIR=examples/demo-apps/android/ExecuTorchDemo/app/src/main/java/com/example/executorchdemo/executor/
  mkdir -p "${EXECUTOR_JAVA_DIR}"
  cp extension/android/src/main/java/org/pytorch/executorch/*.java "${EXECUTOR_JAVA_DIR}"

  NDK_INSTALLATION_DIR=/opt/ndk
  BUCK2=$(which buck2)
  FLATC=$(which flatc)

  rm -rf cmake-out && mkdir cmake-out

  pushd cmake-out
  cmake .. -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_TOOLCHAIN_FILE="${NDK_INSTALLATION_DIR}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DBUCK2="${BUCK2}" \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_FLATC=OFF \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DFLATC_EXECUTABLE="${FLATC}" \
        -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON
  popd
}

build_executorch_android_demo_app
