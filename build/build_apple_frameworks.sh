#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SOURCE_ROOT_DIR=""
OUTPUT="cmake-out"
MODE="Release"
TOOLCHAIN=""
PYTHON=$(which python3)
FLATC=$(which flatc)
COREML=OFF
CUSTOM=OFF
MPS=OFF
OPTIMIZED=OFF
PORTABLE=OFF
QUANTIZED=OFF
XNNPACK=OFF
HEADERS_PATH="include"

PLATFORMS=("ios" "simulator" "macos")
PLATFORM_FLAGS=("OS64" "SIMULATORARM64" "MAC_ARM64")
PLATFORM_TARGET=("17.0" "17.0" "10.15")

FRAMEWORK_EXECUTORCH="executorch:\
libexecutorch.a,\
libexecutorch_no_prim_ops.a,\
libextension_apple.a,\
libextension_data_loader.a,\
libextension_module.a,\
libextension_tensor.a,\
:$HEADERS_PATH"

FRAMEWORK_BACKEND_COREML="backend_coreml:\
libcoremldelegate.a,\
:"

FRAMEWORK_BACKEND_MPS="backend_mps:\
libmpsdelegate.a,\
:"

FRAMEWORK_BACKEND_XNNPACK="backend_xnnpack:\
libXNNPACK.a,\
libcpuinfo.a,\
libpthreadpool.a,\
libxnnpack_backend.a,\
:"

FRAMEWORK_KERNELS_CUSTOM="kernels_custom:\
libcustom_ops.a,\
:"

FRAMEWORK_KERNELS_OPTIMIZED="kernels_optimized:\
liboptimized_kernels.a,\
liboptimized_native_cpu_ops_lib.a,\
:"

FRAMEWORK_KERNELS_PORTABLE="kernels_portable:\
libportable_kernels.a,\
libportable_ops_lib.a,\
:"

FRAMEWORK_KERNELS_QUANTIZED="kernels_quantized:\
libquantized_kernels.a,\
libquantized_ops_lib.a,\
:"

usage() {
  echo "Usage: $0 [SOURCE_ROOT_DIR] [OPTIONS]"
  echo "Build frameworks for Apple platforms."
  echo "SOURCE_ROOT_DIR defaults to the current directory if not provided."
  echo
  echo "Options:"
  echo "  --output=DIR         Output directory. Default: 'cmake-out'"
  echo "  --Debug              Use Debug build mode. Default: Uses Release build mode."
  echo "  --toolchain=FILE     Cmake toolchain file. Default: '\$SOURCE_ROOT_DIR/third-party/ios-cmake/ios.toolchain.cmake'"
  echo "  --python=FILE        Python executable path. Default: Path of python3 found in the current \$PATH"
  echo "  --flatc=FILE         FlatBuffers Compiler executable path. Default: Path of flatc found in the current \$PATH"
  echo "  --coreml             Include this flag to build the Core ML backend."
  echo "  --custom             Include this flag to build the Custom kernels."
  echo "  --mps                Include this flag to build the Metal Performance Shaders backend."
  echo "  --optimized          Include this flag to build the Optimized kernels."
  echo "  --portable           Include this flag to build the Portable kernels."
  echo "  --quantized          Include this flag to build the Quantized kernels."
  echo "  --xnnpack            Include this flag to build the XNNPACK backend."
  echo
  echo "Example:"
  echo "  $0 /path/to/source/root --output=cmake-out --toolchain=/path/to/cmake/toolchain --python=/path/to/python3 --coreml --mps --xnnpack"
  exit 0
}

for arg in "$@"; do
  case $arg in
      -h|--help) usage ;;
      --output=*) OUTPUT="${arg#*=}" ;;
      --Debug) MODE="Debug" ;;
      --toolchain=*) TOOLCHAIN="${arg#*=}" ;;
      --python=*) PYTHON="${arg#*=}" ;;
      --flatc=*) FLATC="${arg#*=}" ;;
      --coreml) COREML=ON ;;
      --custom) CUSTOM=ON ;;
      --mps) MPS=ON ;;
      --optimized) OPTIMIZED=ON ;;
      --portable) PORTABLE=ON ;;
      --quantized) QUANTIZED=ON ;;
      --xnnpack) XNNPACK=ON ;;
      *)
      if [[ -z "$SOURCE_ROOT_DIR" ]]; then
          SOURCE_ROOT_DIR="$arg"
      else
          echo "Invalid argument: $arg"
          exit 1
      fi
      ;;
  esac
done

if [[ -z "$SOURCE_ROOT_DIR" ]]; then
    SOURCE_ROOT_DIR=$(pwd)
fi

if [[ -z "$TOOLCHAIN" ]]; then
    TOOLCHAIN="$SOURCE_ROOT_DIR/third-party/ios-cmake/ios.toolchain.cmake"
fi
[[ -f "$TOOLCHAIN" ]] || { echo >&2 "Toolchain file $TOOLCHAIN does not exist."; exit 1; }

check_command() {
  command -v "$1" >/dev/null 2>&1 || { echo >&2 "$1 is not installed"; exit 1; }
}

check_command cmake
check_command rsync
check_command "$PYTHON"
check_command "$FLATC"

echo "Building libraries"

rm -rf "$OUTPUT" && mkdir -p "$OUTPUT" && cd "$OUTPUT" || exit 1

cmake_build() {
    local platform=$1
    local platform_flag=$2
    local platform_target=$3
    echo "Building for $platform with flag $platform_flag"
    mkdir "$platform" && cd "$platform" || exit 1
    cmake "$SOURCE_ROOT_DIR" -G Xcode \
        -DCMAKE_BUILD_TYPE="$MODE" \
        -DCMAKE_PREFIX_PATH="$($PYTHON -c 'import torch as _; print(_.__path__[0])')" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DCMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD="c++17" \
        -DCMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY="libc++" \
        -DPYTHON_EXECUTABLE="$PYTHON" \
        -DFLATC_EXECUTABLE="$FLATC" \
        -DEXECUTORCH_BUILD_COREML=$COREML \
        -DEXECUTORCH_BUILD_MPS=$MPS \
        -DEXECUTORCH_BUILD_XNNPACK=$XNNPACK \
        -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_APPLE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=$CUSTOM \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=$OPTIMIZED \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=$QUANTIZED \
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$(pwd)" \
        ${platform_flag:+-DPLATFORM=$platform_flag} \
        ${platform_target:+-DDEPLOYMENT_TARGET=$platform_target} \
        --log-level=VERBOSE
    cmake --build . \
        --config $MODE \
        --verbose
    cd ..
}

for index in ${!PLATFORMS[*]}; do
  cmake_build "${PLATFORMS[$index]}" "${PLATFORM_FLAGS[$index]}" "${PLATFORM_TARGET[$index]}"
done

echo "Exporting headers"

mkdir -p "$HEADERS_PATH"

# Set BUCK2 to the path of the buck2 executable in $OUTPUT/*/buck2-bin/buck2-*
BUCK2=$(find . -type f -path '*/buck2-bin/buck2-*' | head -n 1)
if [[ -z "$BUCK2" ]]; then
  echo "Could not find buck2 executable in any buck2-bin directory under $OUTPUT"
  BUCK2=$(which buck2)
fi

check_command "$BUCK2"

"$SOURCE_ROOT_DIR"/build/print_exported_headers.py --buck2=$(realpath "$BUCK2") --targets \
  //extension/module: \
  //extension/tensor: \
| rsync -av --files-from=- "$SOURCE_ROOT_DIR" "$HEADERS_PATH/executorch"

cp "$SOURCE_ROOT_DIR/extension/apple/ExecuTorch/Exported/"*.h "$HEADERS_PATH/executorch"
cp "$SOURCE_ROOT_DIR/extension/apple/ExecuTorch/Exported/"*.modulemap "$HEADERS_PATH"

echo "Creating frameworks"

for platform in "${PLATFORMS[@]}"; do
  echo "Directory: $platform/$MODE"
  FRAMEWORK_FLAGS+=("--directory=$platform/$MODE")
done

append_framework_flag() {
  local flag="$1"
  local framework="$2"
  if [[ $flag == ON ]]; then
    echo "Framework: $framework"
    FRAMEWORK_FLAGS+=("--framework=$framework")
  fi
}

append_framework_flag "ON" "$FRAMEWORK_EXECUTORCH"
append_framework_flag "$COREML" "$FRAMEWORK_BACKEND_COREML"
append_framework_flag "$MPS" "$FRAMEWORK_BACKEND_MPS"
append_framework_flag "$XNNPACK" "$FRAMEWORK_BACKEND_XNNPACK"
append_framework_flag "$CUSTOM" "$FRAMEWORK_KERNELS_CUSTOM"
append_framework_flag "$OPTIMIZED" "$FRAMEWORK_KERNELS_OPTIMIZED"
append_framework_flag "$PORTABLE" "$FRAMEWORK_KERNELS_PORTABLE"
append_framework_flag "$QUANTIZED" "$FRAMEWORK_KERNELS_QUANTIZED"

"$SOURCE_ROOT_DIR"/build/create_frameworks.sh "${FRAMEWORK_FLAGS[@]}"

echo "Cleaning up"

for platform in "${PLATFORMS[@]}"; do
  rm -rf "$platform"
done

rm -rf "$HEADERS_PATH"

echo "Build succeeded!"
