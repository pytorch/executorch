#!/bin/bash
set -eu


export ANDROID_NDK="/opt/android_ndk/r17fb2"
# android_abis=("arm64-v8a" "x86_64")
android_abis=("x86_64")
cmake_jobs=$(( $(sysctl -n hw.ncpu) - 1 ))

for abi in "${android_abis[@]}"; do
    echo "abi: ${abi}"

    out_dir="cmake-out-android-${abi}"
    mkdir -p "${out_dir}"
    echo "out_dir: ${out_dir}"

  cmake . -DCMAKE_INSTALL_PREFIX="${out_dir}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${abi}" \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -B"${out_dir}"

  cmake --build "${out_dir}" -j "${cmake_jobs}" --target install
done
