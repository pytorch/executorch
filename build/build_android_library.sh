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
  cp extension/android/build/libs/executorch.jar "${BUILD_AAR_DIR}/libs"
}

build_android_native_library() {
  ANDROID_ABI="$1"
  ANDROID_NDK="${ANDROID_NDK:-/opt/ndk}"
  CMAKE_OUT="cmake-out-android-$1"
  EXECUTORCH_USE_TIKTOKEN="${EXECUTORCH_USE_TIKTOKEN:-OFF}"
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

  cp "${CMAKE_OUT}"/extension/android/*.so "${BUILD_AAR_DIR}/jni/$1/"
}

build_aar() {
  echo \<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" \
   package=\"org.pytorch.executorch\"\> \
   \<uses-sdk android:minSdkVersion=\"19\" /\> \
   \</manifest\> > "${BUILD_AAR_DIR}/AndroidManifest.xml"
  pushd "${BUILD_AAR_DIR}"
  # Rename libexecutorch_jni.so to libexecutorch.so for soname consistency
  # between Java and JNI
  mv jni/arm64-v8a/libexecutorch_jni.so jni/arm64-v8a/libexecutorch.so
  mv jni/x86_64/libexecutorch_jni.so jni/x86_64/libexecutorch.so
  zip -r executorch.aar libs jni/arm64-v8a/libexecutorch.so jni/x86_64/libexecutorch.so AndroidManifest.xml

  rm jni/arm64-v8a/libexecutorch.so jni/x86_64/libexecutorch.so
  zip -r executorch-llama.aar libs jni/arm64-v8a/libexecutorch_llama_jni.so jni/x86_64/libexecutorch_llama_jni.so AndroidManifest.xml
  popd
  cp "${BUILD_AAR_DIR}/executorch-llama.aar" .
  cp "${BUILD_AAR_DIR}/executorch.aar" .
}

BUILD_AAR_DIR="$(mktemp -d)"
export BUILD_AAR_DIR
mkdir -p "${BUILD_AAR_DIR}/jni/arm64-v8a" "${BUILD_AAR_DIR}/jni/x86_64" "${BUILD_AAR_DIR}/libs"
build_jar
build_android_native_library arm64-v8a
build_android_native_library x86_64
build_aar
