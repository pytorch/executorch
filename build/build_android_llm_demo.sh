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
  ANDROID_NDK="${ANDROID_NDK:-/opt/ndk}"
  CMAKE_OUT="cmake-out-android-${ANDROID_ABI}"
  QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
  if [ -n "$QNN_SDK_ROOT" ]; then
    EXECUTORCH_BUILD_QNN=ON
  else
    EXECUTORCH_BUILD_QNN=OFF
  fi


  cmake . -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_PLATFORM=android-23 \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_LOG_LEVEL=Info \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_QNN="${EXECUTORCH_BUILD_QNN}" \
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
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_LOG_LEVEL=Info \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_LLAMA_JNI=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -B"${CMAKE_OUT}"/extension/android

  cmake --build "${CMAKE_OUT}"/extension/android -j "${CMAKE_JOBS}" --config Release

  # Copy artifacts to ABI specific directory
  mkdir -p "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}"
  cp "${CMAKE_OUT}"/extension/android/*.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"

  # Copy QNN related so library
  if [ -n "$QNN_SDK_ROOT" ] && [ "$ANDROID_ABI" == "arm64-v8a" ]; then
    cp "${CMAKE_OUT}"/lib/libqnn_executorch_backend.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtp.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnSystem.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV69Stub.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV73Stub.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV75Stub.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/"
  fi
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
  zip -r executorch.aar libs jni/*/libexecutorch.so jni/*/libqnn*.so jni/*/libQnn*.so AndroidManifest.xml
  cp executorch.aar executorch-llama.aar
  popd
}

build_android_demo_apps() {
  mkdir -p examples/demo-apps/android/LlamaDemo/app/libs
  cp ${BUILD_AAR_DIR}/executorch-llama.aar examples/demo-apps/android/LlamaDemo/app/libs
  pushd examples/demo-apps/android/LlamaDemo
  ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew build assembleAndroidTest
  popd

  mkdir -p extension/android/benchmark/app/libs
  cp ${BUILD_AAR_DIR}/executorch.aar extension/android/benchmark/app/libs
  pushd extension/android/benchmark
  ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew build assembleAndroidTest
  popd
}

collect_artifacts_to_be_uploaded() {
  ARTIFACTS_DIR_NAME="$1"
  DEMO_APP_DIR="${ARTIFACTS_DIR_NAME}/llm_demo"
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
  # Collect MiniBench APK
  MINIBENCH_APP_DIR="${ARTIFACTS_DIR_NAME}/minibench"
  mkdir -p "${MINIBENCH_APP_DIR}"
  cp extension/android/benchmark/app/build/outputs/apk/debug/*.apk "${MINIBENCH_APP_DIR}"
  cp extension/android/benchmark/app/build/outputs/apk/androidTest/debug/*.apk "${MINIBENCH_APP_DIR}"
}

BUILD_AAR_DIR="$(mktemp -d)"
export BUILD_AAR_DIR
if [ -z "$ANDROID_ABIS" ]; then
  ANDROID_ABIS=("arm64-v8a" "x86_64")
fi
export ANDROID_ABIS

ARTIFACTS_DIR_NAME="$1"

build_jar
for ANDROID_ABI in "${ANDROID_ABIS[@]}"; do
  build_android_native_library ${ANDROID_ABI}
done
build_aar
build_android_demo_apps
collect_artifacts_to_be_uploaded ${ARTIFACTS_DIR_NAME}
