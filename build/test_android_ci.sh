#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# https://github.com/pytorch/executorch/tree/main/examples/demo-apps/android/ExecuTorchDemo
build_executorch() {
  MODEL_NAME=dl3
  # Delegating DeepLab v3 to XNNPACK backend
  python -m examples.xnnpack.aot_compiler --model_name="${MODEL_NAME}" --delegate

  ASSETS_DIR=examples/demo-apps/android/ExecuTorchDemo/app/src/main/assets/
  ANDROID_NDK_VERSION=android-ndk-r26c
  ANDROID_NDK_URL=https://dl.google.com/android/repository/
  DOWNLOAD_LINK="${ANDROID_NDK_URL}${ANDROID_NDK_VERSION}-linux.zip"
  pushd /tmp/
  wget ${DOWNLOAD_LINK}
  unzip "${ANDROID_NDK_VERSION}-linux.zip"
  popd

  mkdir -p "${ASSETS_DIR}"
  cp "${MODEL_NAME}_xnnpack_fp32.pte" "${ASSETS_DIR}"

  rm -rf cmake-out && mkdir cmake-out
  ANDROID_NDK="/tmp/${ANDROID_NDK_VERSION}" BUCK2=/home/kimishpatel/buck2 FLATC=$(which flatc) ANDROID_ABI=arm64-v8a \
    bash examples/demo-apps/android/ExecuTorchDemo/setup.sh
}

build_android_demo_app() {
  pushd examples/demo-apps/android/ExecuTorchDemo
  ANDROID_HOME=/opt/android/sdk ./gradlew build
  popd
}

build_executorch
build_android_demo_app
