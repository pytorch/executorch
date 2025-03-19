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

copy_src() {
  cp -r extension/android/build.gradle extension/android/settings.gradle extension/android/gradlew extension/android/gradle extension/android/gradlew.bat "${BUILD_AAR_DIR}"
  cp -r extension/android/executorch_android "${BUILD_AAR_DIR}/executorch_android"
}

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
    -DCMAKE_BUILD_TYPE="${EXECUTORCH_CMAKE_BUILD_TYPE}" \
    -B"${CMAKE_OUT}"

  if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
  else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
  fi
  cmake --build "${CMAKE_OUT}" -j "${CMAKE_JOBS}" --target install --config "${EXECUTORCH_CMAKE_BUILD_TYPE}"

  # Update tokenizers submodule
  pushd extension/llm/tokenizers
  echo "Update tokenizers submodule"
  git submodule update --init
  popd
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
    -DCMAKE_BUILD_TYPE="${EXECUTORCH_CMAKE_BUILD_TYPE}" \
    -B"${CMAKE_OUT}"/extension/android

  cmake --build "${CMAKE_OUT}"/extension/android -j "${CMAKE_JOBS}" --config "${EXECUTORCH_CMAKE_BUILD_TYPE}"

  # Copy artifacts to ABI specific directory
  mkdir -p "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}"
  cp "${CMAKE_OUT}"/extension/android/*.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"

  # Copy QNN related so library
  if [ -n "$QNN_SDK_ROOT" ] && [ "$ANDROID_ABI" == "arm64-v8a" ]; then
    cp "${CMAKE_OUT}"/lib/libqnn_executorch_backend.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtp.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnSystem.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV69Stub.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV73Stub.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/aarch64-android/libQnnHtpV75Stub.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"
    cp "${QNN_SDK_ROOT}"/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so "${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/"
  fi

  # Copy MTK related so library
  if [ -n "$NEURON_BUFFER_ALLOCATOR_LIB" ] && [ -n "$NEURON_USDK_ADAPTER_LIB" ] && [ "$ANDROID_ABI" == "arm64-v8a" ]; then
    cp "${CMAKE_OUT}"/backends/mediatek/libneuron_backend.so ${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/
    cp "${NEURON_BUFFER_ALLOCATOR_LIB}" ${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/
    cp "${NEURON_USDK_ADAPTER_LIB}" ${BUILD_AAR_DIR}/executorch_android/src/main/jniLibs/${ANDROID_ABI}/
  fi
}

build_aar() {
  pushd "${BUILD_AAR_DIR}"
  # Rename libexecutorch_jni.so to libexecutorch.so for soname consistency
  # between Java and JNI
  find . -type f -name "libexecutorch_jni.so" -exec bash -c 'mv "$1" "${1/_jni/}"' bash {} \;
  if [ "$EXECUTORCH_CMAKE_BUILD_TYPE" == "Release" ]; then
    find . -type f -name "*.so" -exec "$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip {} \;
  fi
  ./gradlew build
  cp executorch_android/build/outputs/aar/executorch_android-debug.aar executorch.aar
  popd
}

main() {
  if [[ -z "${BUILD_AAR_DIR:-}" ]]; then
    BUILD_AAR_DIR="$(mktemp -d)"
  fi
  export BUILD_AAR_DIR
  if [ -z "$ANDROID_ABIS" ]; then
    ANDROID_ABIS=("arm64-v8a" "x86_64")
  fi
  export ANDROID_ABIS

  ARTIFACTS_DIR_NAME="$1"

  copy_src
  for ANDROID_ABI in "${ANDROID_ABIS[@]}"; do
    build_android_native_library ${ANDROID_ABI}
  done
  build_aar
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
