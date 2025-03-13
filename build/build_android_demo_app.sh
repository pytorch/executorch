#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


build_android_llama_demo() {
  mkdir -p examples/demo-apps/android/LlamaDemo/app/libs
  cp ${BUILD_AAR_DIR}/executorch.aar examples/demo-apps/android/LlamaDemo/app/libs
  pushd examples/demo-apps/android/LlamaDemo
  ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew build assembleAndroidTest
  popd
}

build_android_benchmark_app() {
  mkdir -p extension/benchmark/android/benchmark/app/libs
  cp ${BUILD_AAR_DIR}/executorch.aar extension/benchmark/android/benchmark/app/libs
  pushd extension/benchmark/android/benchmark
  ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew build assembleAndroidTest
  popd
}

build_android_test() {
  pushd extension/android_test
  ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew testDebugUnitTest
  ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew build assembleAndroidTest
  popd
}


if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  build_android_llama_demo
  build_android_benchmark_app
  build_android_test
fi
