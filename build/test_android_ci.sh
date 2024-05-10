#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# https://github.com/pytorch/executorch/tree/main/examples/demo-apps/android/ExecuTorchDemo
export_model() {
  MODEL_NAME=dl3
  # Delegating DeepLab v3 to XNNPACK backend
  python -m examples.xnnpack.aot_compiler --model_name="${MODEL_NAME}" --delegate

  ASSETS_DIR=examples/demo-apps/android/ExecuTorchDemo/app/src/main/assets/
  mkdir -p "${ASSETS_DIR}"
  cp "${MODEL_NAME}_xnnpack_fp32.pte" "${ASSETS_DIR}"
}

build_android_native_library() {
  pushd examples/demo-apps/android/LlamaDemo
  CMAKE_OUT="cmake-out-android-$1" ANDROID_NDK=/opt/ndk ANDROID_ABI="$1" ./gradlew setup
  popd
  cp "cmake-out-android-$1"/extension/android/*.so build_aar/jni/$1/
}

build_android_demo_app() {
  pushd examples/demo-apps/android/ExecuTorchDemo
  ANDROID_HOME=/opt/android/sdk ./gradlew build
  popd
}

build_android_llama_demo_app() {
  pushd examples/demo-apps/android/LlamaDemo
  ANDROID_HOME=/opt/android/sdk ./gradlew build
  ANDROID_HOME=/opt/android/sdk ./gradlew assembleAndroidTest
  popd
}

build_aar() {
  cp extension/android/build/libs/executorch.jar build_aar/libs
  echo \<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" \
   package=\"org.pytorch.executorch\"\> \
   \<uses-sdk android:minSdkVersion=\"19\" /\> \
   \</manifest\> > build_aar/AndroidManifest.xml
  pushd build_aar
  zip -r executorch.aar libs jni AndroidManifest.xml

  rm jni/arm64-v8a/libexecutorch_jni.so jni/x86_64/libexecutorch_jni.so
  zip -r executorch-llama.aar libs jni AndroidManifest.xml
  popd
}

mkdir -p build_aar/jni/arm64-v8a build_aar/jni/x86_64 build_aar/libs

build_android_native_library arm64-v8a
build_android_native_library x86_64
export_model
build_android_demo_app
build_android_llama_demo_app
build_aar
