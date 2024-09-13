#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

CMAKE_OUT="${CMAKE_OUT:-cmake-out-android}"
# Note: Set up ANDROID_NDK and ANDROID_ABI
cmake . -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_QNN=ON \
  -DQNN_SDK_ROOT="${QNN_SDK_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -B"${CMAKE_OUT}"

if [ "$(uname)" == "Darwin" ]; then
  CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
  CMAKE_JOBS=$(( $(nproc) - 1 ))
fi
cmake --build "${CMAKE_OUT}" -j "${CMAKE_JOBS}" --target install --config Release

cmake extension/android \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
  -DEXECUTORCH_BUILD_LLAMA_JNI=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -B"${CMAKE_OUT}"/extension/android

cmake --build "${CMAKE_OUT}"/extension/android -j "${CMAKE_JOBS}" --config Release

JNI_LIBS_PATH="examples/demo-apps/android/LlamaDemo/app/src/main/jniLibs"
mkdir -p "${JNI_LIBS_PATH}/${ANDROID_ABI}"

BUILD_AAR_DIR="$(mktemp -d)"
mkdir -p "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}" "${BUILD_AAR_DIR}/libs"
JNI_LIBS_PATH="${BUILD_AAR_DIR}/jni"
cp "${CMAKE_OUT}"/extension/android/libexecutorch_jni.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/libexecutorch.so"
cp "${CMAKE_OUT}"/lib/libqnn_executorch_backend.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtp.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnSystem.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV69Stub.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV73Stub.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV75Stub.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
cp "${QNN_SDK_ROOT}"/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
cp "${QNN_SDK_ROOT}"/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
cp "${QNN_SDK_ROOT}"/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
cp extension/android/build/libs/executorch.jar "${BUILD_AAR_DIR}/libs"
echo \<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" \
  package=\"org.pytorch.executorch\"\> \
  \<uses-sdk android:minSdkVersion=\"19\" /\> \
  \</manifest\> > "${BUILD_AAR_DIR}/AndroidManifest.xml"
pushd "${BUILD_AAR_DIR}"
zip -r executorch-llama.aar libs jni/${ANDROID_ABI} AndroidManifest.xml
popd
mkdir -p examples/demo-apps/android/LlamaDemo/app/libs
mv "${BUILD_AAR_DIR}/executorch-llama.aar" examples/demo-apps/android/LlamaDemo/app/libs
