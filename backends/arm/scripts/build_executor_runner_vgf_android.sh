#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build the executor_runner application for the Arm VGF backend for
# deployment on an Android device.
#
# Usage:
#   export ANDROID_HOME=/path/to/Android/sdk
#   export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/<version>
#   ./backends/arm/scripts/build_executor_runner_vgf_android.sh
#

set -eu

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/utils.sh"

et_root_dir=$(cd "${script_dir}/../../.." && pwd)
mlsdk_dir="${et_root_dir}/examples/arm/arm-scratch/ml-sdk-for-vulkan-manifest"
output_dir="${et_root_dir}/cmake-out-vgf-android"
build_type=Release
android_abi=arm64-v8a
android_platform=android-28
build_with_etdump=false
build_devtools=OFF
build_event_tracer=OFF

help() {
  echo "Usage: $(basename "$0") [options]"
  echo "Note: this developer build script is not a stable public API."
  echo "Options:"
  echo "  --output=<DIR>             Build directory (default: ${output_dir})"
  echo "  --build-type=<TYPE>        CMake build type (default: ${build_type})"
  echo "  --android-abi=<ABI>        Android ABI (default: ${android_abi})"
  echo "  --android-platform=<API>   Android platform (default: ${android_platform})"
  echo "  --etdump                   Build with ETDump profiling support"
}

for arg in "$@"; do
  case ${arg} in
    -h | --help)
      help
      exit 0
      ;;
    --output=*) output_dir="${arg#*=}" ;;
    --build-type=*) build_type="${arg#*=}" ;;
    --android-abi=*) android_abi="${arg#*=}" ;;
    --android-platform=*) android_platform="${arg#*=}" ;;
    --etdump) build_with_etdump=true ;;
    *)
      echo "Unknown option: ${arg}" >&2
      help >&2
      exit 1
      ;;
  esac
done

android_ndk=${ANDROID_NDK_HOME:-${ANDROID_NDK:-}}
if [[ -z ${android_ndk} ]]; then
  echo "Set ANDROID_NDK_HOME or ANDROID_NDK to the Android NDK directory." >&2
  exit 1
fi

toolchain_file="${android_ndk}/build/cmake/android.toolchain.cmake"
vgf_source_dir="${mlsdk_dir}/sw/vgf-lib"
flatbuffers_dir="${mlsdk_dir}/dependencies/flatbuffers"
vgf_build_dir="${output_dir}/vgf-lib"
vgf_install_dir="${output_dir}/vgf-install"
executorch_build_dir="${output_dir}/executorch"

[[ -f ${toolchain_file} ]] || {
  echo "Android CMake toolchain not found: ${toolchain_file}" >&2
  exit 1
}
[[ -f ${vgf_source_dir}/CMakeLists.txt ]] || {
  echo "VGF source not found: ${vgf_source_dir}" >&2
  echo "Install the MLSDK source tree with examples/arm/setup.sh first." >&2
  exit 1
}
[[ -f ${flatbuffers_dir}/CMakeLists.txt ]] || {
  echo "MLSDK FlatBuffers source not found: ${flatbuffers_dir}" >&2
  exit 1
}

mkdir -p "${output_dir}"

if [[ ${build_with_etdump} == true ]]; then
  build_devtools=ON
  build_event_tracer=ON
fi

# Cross-compile the VGF library for Android
cmake \
  -S "${vgf_source_dir}" \
  -B "${vgf_build_dir}" \
  -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}" \
  -DANDROID_ABI="${android_abi}" \
  -DANDROID_PLATFORM="${android_platform}" \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_INSTALL_PREFIX="${vgf_install_dir}" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DFLATBUFFERS_PATH="${flatbuffers_dir}" \
  -DML_SDK_VGF_LIB_BUILD_TOOLS=OFF \
  -DML_SDK_VGF_LIB_BUILD_TESTS=OFF \
  -DML_SDK_VGF_LIB_BUILD_PYLIB=OFF \
  -DML_SDK_VGF_LIB_BUILD_SHARED=OFF

parallel_jobs="$(get_parallel_jobs)"

cmake --build "${vgf_build_dir}" --target vgf --parallel "${parallel_jobs}"
cmake --install "${vgf_build_dir}"

# Cross-compile ExecuTorch for Android and link against the Android build of the libvgf.a
cmake \
  -S "${et_root_dir}" \
  -B "${executorch_build_dir}" \
  -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}" \
  -DANDROID_ABI="${android_abi}" \
  -DANDROID_PLATFORM="${android_platform}" \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DEXECUTORCH_PAL_DEFAULT=android \
  -DEXECUTORCH_VGF_ROOT="${vgf_install_dir}" \
  -DEXECUTORCH_BUILD_VGF=ON \
  -DEXECUTORCH_BUILD_VULKAN=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_DEVTOOLS="${build_devtools}" \
  -DEXECUTORCH_ENABLE_EVENT_TRACER="${build_event_tracer}"

cmake --build "${executorch_build_dir}" \
  --target executor_runner \
  --parallel "${parallel_jobs}"

echo "Android VGF executor runner: ${executorch_build_dir}/executor_runner"
