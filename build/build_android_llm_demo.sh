#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

build_jar() {
  pushd extension/android
  ./gradlew build
  popd
  mkdir -p "${BUILD_AAR_DIR}/libs"
  cp extension/android/build/libs/executorch.jar "${BUILD_AAR_DIR}/libs/"
}

build_android_native_library() {
  ANDROID_ABI="$1"
  TOKENIZER="$2"
  ANDROID_NDK="${ANDROID_NDK:-/opt/ndk}"
  CMAKE_OUT="cmake-out-android-${ANDROID_ABI}"
  if [[ $TOKENIZER == "tiktoken" ]]; then
    EXECUTORCH_USE_TIKTOKEN=ON
  else
    EXECUTORCH_USE_TIKTOKEN=OFF
  fi

  cmake . -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_PLATFORM=android-23 \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -B"${CMAKE_OUT}"

  if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
  else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
  fi
  cmake --build "${CMAKE_OUT}" -j "${CMAKE_JOBS}" --target install --config Release

  cmake examples/models/llama2 \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -DEXECUTORCH_USE_TIKTOKEN="${EXECUTORCH_USE_TIKTOKEN}" \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -B"${CMAKE_OUT}"/examples/models/llama2

  cmake --build "${CMAKE_OUT}"/examples/models/llama2 -j "${CMAKE_JOBS}" --config Release

  cmake extension/android \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -DEXECUTORCH_BUILD_LLAMA_JNI=ON \
    -DEXECUTORCH_USE_TIKTOKEN="${EXECUTORCH_USE_TIKTOKEN}" \
    -DCMAKE_BUILD_TYPE=Release \
    -B"${CMAKE_OUT}"/extension/android

  cmake --build "${CMAKE_OUT}"/extension/android -j "${CMAKE_JOBS}" --config Release

  # Copy artifacts to ABI specific directory
  mkdir -p "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}"
  cp "${CMAKE_OUT}"/extension/android/*.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
}

build_aar() {
  echo \<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" \
  package=\"org.pytorch.executorch\"\> \
  \<uses-sdk android:minSdkVersion=\"19\" /\> \
  \</manifest\> > "${BUILD_AAR_DIR}/AndroidManifest.xml"
  pushd "${BUILD_AAR_DIR}"
  # Rename libexecutorch_jni.so to libexecutorch.so for soname consistency
  # between Java and JNI
  find jni -type f -name "libexecutorch_jni.so" -exec bash -c 'mv "$1" "${1/_jni/}"' bash {} \;
  # Zip all necessary files into the AAR file
  zip -r executorch.aar libs jni/*/libexecutorch.so AndroidManifest.xml
  zip -r executorch-llama.aar libs jni/*/libexecutorch_llama_jni.so AndroidManifest.xml
  popd
}

build_android_llm_demo_app() {
  mkdir -p examples/demo-apps/android/LlamaDemo/app/libs
  cp ${BUILD_AAR_DIR}/executorch-llama.aar examples/demo-apps/android/LlamaDemo/app/libs
  pushd examples/demo-apps/android/LlamaDemo
  ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew build assembleAndroidTest
  popd
}

collect_artifacts_to_be_uploaded() {
  TOKENIZER="$1"
  ARTIFACTS_DIR_NAME="$2"
  DEMO_APP_DIR="${ARTIFACTS_DIR_NAME}/llm_demo_${TOKENIZER}"
  # The app directory is named using its build flavor as a suffix.
  mkdir -p "${DEMO_APP_DIR}"
  # Collect the app and its test suite
  cp examples/demo-apps/android/LlamaDemo/app/build/outputs/apk/debug/*.apk "${DEMO_APP_DIR}"
  cp examples/demo-apps/android/LlamaDemo/app/build/outputs/apk/androidTest/debug/*.apk "${DEMO_APP_DIR}"
  # Collect all ABI specific libraries
  for ANDROID_ABI in "${ANDROID_ABIS[@]}"; do
    mkdir -p "${DEMO_APP_DIR}/${ANDROID_ABI}"
    cp cmake-out-android-${ANDROID_ABI}/lib/*.a "${DEMO_APP_DIR}/${ANDROID_ABI}/"
    cp cmake-out-android-${ANDROID_ABI}/extension/android/*.so "${DEMO_APP_DIR}/${ANDROID_ABI}/"
  done
  # Collect JAR and AAR
  cp extension/android/build/libs/executorch.jar "${DEMO_APP_DIR}"
  find "${BUILD_AAR_DIR}/" -name 'executorch*.aar' -exec cp {} "${DEMO_APP_DIR}" \;
}

BUILD_AAR_DIR="$(mktemp -d)"
export BUILD_AAR_DIR
ANDROID_ABIS=("arm64-v8a" "x86_64")
export ANDROID_ABIS

TOKENIZER="${1:-bpe}"
ARTIFACTS_DIR_NAME="$2"

build_jar
for ANDROID_ABI in "${ANDROID_ABIS[@]}"; do
  build_android_native_library ${ANDROID_ABI} ${TOKENIZER}
done
build_aar
build_android_llm_demo_app
collect_artifacts_to_be_uploaded ${TOKENIZER} ${ARTIFACTS_DIR_NAME}
