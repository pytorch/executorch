#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi
which "${PYTHON_EXECUTABLE}"

build_android_native_library() {
  ANDROID_ABI="$1"
  ANDROID_NDK="${ANDROID_NDK:-/opt/ndk}"
  CMAKE_OUT="cmake-out-android-${ANDROID_ABI}"
  EXECUTORCH_CMAKE_BUILD_TYPE="${EXECUTORCH_CMAKE_BUILD_TYPE:-Release}"
  QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
  if [ -n "$QNN_SDK_ROOT" ]; then
    EXECUTORCH_BUILD_QNN=ON
  else
    EXECUTORCH_BUILD_QNN=OFF
  fi

  NEURON_BUFFER_ALLOCATOR_LIB="${NEURON_BUFFER_ALLOCATOR_LIB:-}"
  NEURON_USDK_ADAPTER_LIB="${NEURON_USDK_ADAPTER_LIB:-}"
  if [ -n "$NEURON_BUFFER_ALLOCATOR_LIB" ]; then
    EXECUTORCH_BUILD_NEURON=ON
  else
    EXECUTORCH_BUILD_NEURON=OFF
  fi

  EXECUTORCH_BUILD_VULKAN="${EXECUTORCH_BUILD_VULKAN:-OFF}"

  cmake . -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_PLATFORM=android-26 \
    -DBUILD_TESTING=OFF \
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
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM="${EXECUTORCH_BUILD_EXTENSION_LLM:-ON}" \
    -DEXECUTORCH_BUILD_NEURON="${EXECUTORCH_BUILD_NEURON}" \
    -DNEURON_BUFFER_ALLOCATOR_LIB="${NEURON_BUFFER_ALLOCATOR_LIB}" \
    -DEXECUTORCH_BUILD_QNN="${EXECUTORCH_BUILD_QNN}" \
    -DQNN_SDK_ROOT="${QNN_SDK_ROOT}" \
    -DEXECUTORCH_BUILD_VULKAN="${EXECUTORCH_BUILD_VULKAN}" \
    -DSUPPORT_REGEX_LOOKAHEAD=ON \
    -DCMAKE_BUILD_TYPE="${EXECUTORCH_CMAKE_BUILD_TYPE}" \
    -B"${CMAKE_OUT}"

  if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
  else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
  fi
  cmake --build "${CMAKE_OUT}" -j "${CMAKE_JOBS}" --target install --config "${EXECUTORCH_CMAKE_BUILD_TYPE}"

  cmake extension/android \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_PLATFORM=android-26 \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_LOG_LEVEL=Info \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -DNEURON_BUFFER_ALLOCATOR_LIB="$NEURON_BUFFER_ALLOCATOR_LIB" \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM="${EXECUTORCH_BUILD_EXTENSION_LLM:-ON}" \
    -DEXECUTORCH_BUILD_LLAMA_JNI="${EXECUTORCH_BUILD_EXTENSION_LLM:-ON}" \
    -DSUPPORT_REGEX_LOOKAHEAD=ON \
    -DCMAKE_BUILD_TYPE="${EXECUTORCH_CMAKE_BUILD_TYPE}" \
    -B"${CMAKE_OUT}"/extension/android

  cmake --build "${CMAKE_OUT}"/extension/android -j "${CMAKE_JOBS}" --config "${EXECUTORCH_CMAKE_BUILD_TYPE}"

  # Copy artifacts to ABI specific directory
  local SO_STAGE_DIR="cmake-out-android-so/${ANDROID_ABI}"
  mkdir -p ${SO_STAGE_DIR}
  cp "${CMAKE_OUT}"/extension/android/*.so "${SO_STAGE_DIR}/libexecutorch.so"

  # Copy QNN related so library
  if [ -n "$QNN_SDK_ROOT" ] && [ "$ANDROID_ABI" == "arm64-v8a" ]; then
    cp "${CMAKE_OUT}"/lib/libqnn_executorch_backend.so ${SO_STAGE_DIR}
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtp.so ${SO_STAGE_DIR}
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnSystem.so ${SO_STAGE_DIR}
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV69Stub.so ${SO_STAGE_DIR}
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV73Stub.so ${SO_STAGE_DIR}
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV75Stub.so ${SO_STAGE_DIR}
    cp "${QNN_SDK_ROOT}"/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so ${SO_STAGE_DIR}
    cp "${QNN_SDK_ROOT}"/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ${SO_STAGE_DIR}
    cp "${QNN_SDK_ROOT}"/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so ${SO_STAGE_DIR}
  fi

  # Copy MTK related so library
  if [ -n "$NEURON_BUFFER_ALLOCATOR_LIB" ] && [ -n "$NEURON_USDK_ADAPTER_LIB" ] && [ "$ANDROID_ABI" == "arm64-v8a" ]; then
    cp "${CMAKE_OUT}"/backends/mediatek/libneuron_backend.so ${SO_STAGE_DIR}
    cp "${NEURON_BUFFER_ALLOCATOR_LIB}" ${SO_STAGE_DIR}
    cp "${NEURON_USDK_ADAPTER_LIB}" ${SO_STAGE_DIR}
  fi
}

build_aar() {
  if [ "$EXECUTORCH_CMAKE_BUILD_TYPE" == "Release" ]; then
    find cmake-out-android-so -type f -name "*.so" -exec "$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip {} \;
  fi
  pushd extension/android/
  ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew build
  # Use java unit test as sanity check
  ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew :executorch_android:testDebugUnitTest
  popd
  if [ ! -z $BUILD_AAR_DIR ]; then
    cp extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar "${BUILD_AAR_DIR}/executorch.aar"
  fi
}

main() {
  if [ -z "$ANDROID_ABIS" ]; then
    ANDROID_ABIS=("arm64-v8a" "x86_64")
  fi
  export ANDROID_ABIS

  mkdir -p cmake-out-android-so/
  for ANDROID_ABI in "${ANDROID_ABIS[@]}"; do
    build_android_native_library ${ANDROID_ABI}
  done
  build_aar
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
