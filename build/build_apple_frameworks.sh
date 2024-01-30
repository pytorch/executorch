#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

PLATFORMS=("iphoneos" "iphonesimulator")
PLATFORM_FLAGS=("OS" "SIMULATOR")
SOURCE_ROOT_DIR=""
OUTPUT="cmake-out"
MODE="Debug"
TOOLCHAIN=""
BUCK2="/tmp/buck2"
PYTHON=$(which python3)
FLATC=""
IOS_DEPLOYMENT_TARGET="17.0"
COREML=OFF
MPS=OFF
XNNPACK=OFF
HEADERS_PATH="include"
EXECUTORCH_FRAMEWORK="executorch:libexecutorch.a,libextension_data_loader.a,libextension_module.a:$HEADERS_PATH"
PORTABLE_FRAMEWORK="portable_backend:libportable_kernels.a,libportable_ops_lib.a:"
COREML_FRAMEWORK="coreml_backend:libcoremldelegate.a:"
MPS_FRAMEWORK="mps_backend:libmpsdelegate.a:"
XNNPACK_FRAMEWORK="xnnpack_backend:libXNNPACK.a,libcpuinfo.a,libpthreadpool.a,libxnnpack_backend.a:"

usage() {
  echo "Usage: $0 [SOURCE_ROOT_DIR] [OPTIONS]"
  echo "Build frameworks for Apple platforms."
  echo "SOURCE_ROOT_DIR defaults to the current directory if not provided."
  echo
  echo "Options:"
  echo "  --output=DIR         Output directory. Default: 'cmake-out'"
  echo "  --Release            Use Release build mode. Default: 'Debug'"
  echo "  --toolchain=FILE     Cmake toolchain file. Default: '\$SOURCE_ROOT_DIR/third-party/pytorch/cmake/iOS.cmake'"
  echo "  --buck2=FILE         Buck2 executable path. Default: '/tmp/buck2'"
  echo "  --python=FILE        Python executable path. Default: Path of python3 found in the current \$PATH"
  echo "  --flatc=FILE         FlatBuffers Compiler executable path. Default: '\$SOURCE_ROOT_DIR/third-party/flatbuffers/cmake-out/flatc'"
  echo "  --coreml             Include this flag to build Core ML backend."
  echo "  --mps                Include this flag to build Metal Performance Shaders backend."
  echo "  --xnnpack            Include this flag to build XNNPACK backend."
  echo
  echo "Example:"
  echo "  $0 /path/to/source/root --output=cmake-out --Release --toolchain=/path/to/cmake/toolchain --buck2=/path/to/buck2 --python=/path/to/python3 --coreml --mps --xnnpack"
  exit 0
}

for arg in "$@"; do
  case $arg in
      -h|--help) usage ;;
      --output=*) OUTPUT="${arg#*=}" ;;
      --Release) MODE="Release" ;;
      --toolchain=*) TOOLCHAIN="${arg#*=}" ;;
      --buck2=*) BUCK2="${arg#*=}" ;;
      --python=*) PYTHON="${arg#*=}" ;;
      --flatc=*) FLATC="${arg#*=}" ;;
      --ios-deployment-target=*) IOS_DEPLOYMENT_TARGET="${arg#*=}" ;;
      --coreml) COREML=ON ;;
      --mps) MPS=ON ;;
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
    TOOLCHAIN="$SOURCE_ROOT_DIR/third-party/pytorch/cmake/iOS.cmake"
fi
[[ -f "$TOOLCHAIN" ]] || { echo >&2 "Toolchain file $TOOLCHAIN does not exist."; exit 1; }

if [[ -z "$FLATC" ]]; then
    FLATC="$SOURCE_ROOT_DIR/third-party/flatbuffers/cmake-out/flatc"
fi

check_command() {
  command -v "$1" >/dev/null 2>&1 || { echo >&2 "$1 is not installed"; exit 1; }
}

check_command cmake
check_command rsync
check_command "$BUCK2"
check_command "$PYTHON"
check_command "$FLATC"

echo "Building libraries"

rm -rf "$OUTPUT" && mkdir -p "$OUTPUT" && cd "$OUTPUT" || exit 1

cmake_build() {
    local platform=$1
    local platform_flag=$2
    echo "Building for $platform with flag $platform_flag"
    mkdir "$platform" && cd "$platform" || exit 1
    cmake "$SOURCE_ROOT_DIR" -G Xcode \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DCMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD="c++17" \
        -DCMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY="libc++" \
        -DBUCK2="$BUCK2" \
        -DPYTHON_EXECUTABLE="$PYTHON" \
        -DFLATC_EXECUTABLE="$FLATC" \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$(pwd)" \
        -DIOS_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
        -DEXECUTORCH_BUILD_COREML=$COREML \
        -DEXECUTORCH_BUILD_MPS=$MPS \
        -DEXECUTORCH_BUILD_XNNPACK=$XNNPACK \
        ${platform_flag:+-DIOS_PLATFORM=$platform_flag}
    cmake --build . --config $MODE
    cd ..
}

for index in ${!PLATFORMS[*]}; do
  cmake_build "${PLATFORMS[$index]}" "${PLATFORM_FLAGS[$index]}"
done

echo "Exporting headers"

mkdir -p "$HEADERS_PATH"

"$SOURCE_ROOT_DIR"/build/print_exported_headers.py --buck2="$BUCK2" --targets \
  //extension/module: \
| rsync -av --files-from=- "$SOURCE_ROOT_DIR" "$HEADERS_PATH/executorch"

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

append_framework_flag "ON" "$EXECUTORCH_FRAMEWORK"
append_framework_flag "ON" "$PORTABLE_FRAMEWORK"
append_framework_flag "$COREML" "$COREML_FRAMEWORK"
append_framework_flag "$MPS" "$MPS_FRAMEWORK"
append_framework_flag "$XNNPACK" "$XNNPACK_FRAMEWORK"

"$SOURCE_ROOT_DIR"/build/create_frameworks.sh "${FRAMEWORK_FLAGS[@]}"

echo "Cleaning up"

for platform in "${PLATFORMS[@]}"; do
  rm -rf "$platform"
done

rm -rf "$HEADERS_PATH"

echo "Build succeeded!"
