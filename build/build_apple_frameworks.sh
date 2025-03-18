#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SOURCE_ROOT_DIR=""
OUTPUT="cmake-out"
MODES=()
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
libexecutorch_core.a,\
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
libextension_threadpool.a,\
libpthreadpool.a,\
libxnnpack_backend.a,\
libmicrokernels-prod.a,\
:"

FRAMEWORK_KERNELS_CUSTOM="kernels_custom:\
libcustom_ops.a,\
:"

FRAMEWORK_KERNELS_OPTIMIZED="kernels_optimized:\
libcpublas.a,\
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
  echo "  --Debug              Build Debug version."
  echo "  --Release            Build Release version."
  echo "  --toolchain=FILE     CMake toolchain file. Default: '\$SOURCE_ROOT_DIR/third-party/ios-cmake/ios.toolchain.cmake'"
  echo "  --python=FILE        Python executable path. Default: Path of python3 in \$PATH"
  echo "  --flatc=FILE         FlatBuffers Compiler executable path. Default: Path of flatc in \$PATH"
  echo "  --coreml             Build the Core ML backend."
  echo "  --custom             Build the Custom kernels."
  echo "  --mps                Build the Metal Performance Shaders backend."
  echo "  --optimized          Build the Optimized kernels."
  echo "  --portable           Build the Portable kernels."
  echo "  --quantized          Build the Quantized kernels."
  echo "  --xnnpack            Build the XNNPACK backend."
  echo
  echo "Example:"
  echo "  $0 /path/to/source/root --output=cmake-out --toolchain=/path/to/toolchain --python=/path/to/python3 --coreml --mps --xnnpack"
  exit 0
}

for arg in "$@"; do
  case $arg in
      -h|--help) usage ;;
      --output=*) OUTPUT="${arg#*=}" ;;
      --Release)
        if [[ ! " ${MODES[*]:-} " =~ \bRelease\b ]]; then
          MODES+=("Release")
        fi
        ;;
      --Debug)
        if [[ ! " ${MODES[*]:-} " =~ \bDebug\b ]]; then
          MODES+=("Debug")
        fi
        ;;
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

if [ ${#MODES[@]} -eq 0 ]; then
  MODES=("Release")
fi

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
    local mode=$4
    echo "Building for $platform ($mode) with flag $platform_flag"
    mkdir -p "$platform" && cd "$platform" || exit 1
    cmake "$SOURCE_ROOT_DIR" -G Xcode \
        -DCMAKE_BUILD_TYPE="$mode" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DCMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD="c++17" \
        -DCMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY="libc++" \
        -DCMAKE_C_FLAGS="-ffile-prefix-map=$SOURCE_ROOT_DIR=/executorch -fdebug-prefix-map=$SOURCE_ROOT_DIR=/executorch" \
        -DCMAKE_CXX_FLAGS="-ffile-prefix-map=$SOURCE_ROOT_DIR=/executorch -fdebug-prefix-map=$SOURCE_ROOT_DIR=/executorch" \
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
        --config "$mode" \
        --verbose
    cd ..
}

for index in ${!PLATFORMS[*]}; do
  for mode in "${MODES[@]}"; do
    cmake_build "${PLATFORMS[$index]}" "${PLATFORM_FLAGS[$index]}" "${PLATFORM_TARGET[$index]}" "$mode"
  done
done

echo "Exporting headers"

mkdir -p "$HEADERS_PATH"

BUCK2=$("$PYTHON" "$SOURCE_ROOT_DIR/tools/cmake/resolve_buck.py" --cache_dir="$SOURCE_ROOT_DIR/buck2-bin")
if [[ -z "$BUCK2" ]]; then
  echo "Could not find buck2 executable in any buck2-bin directory under $SOURCE_ROOT_DIR"
  BUCK2=$(which buck2)
fi

check_command "$BUCK2"

"$SOURCE_ROOT_DIR"/build/print_exported_headers.py --buck2=$(realpath "$BUCK2") --targets \
  //extension/module: \
  //extension/tensor: \
| rsync -av --files-from=- "$SOURCE_ROOT_DIR" "$HEADERS_PATH/executorch"

# HACK: XCFrameworks don't appear to support exporting any build
# options, but we need the following:
# - runtime/core/portable/type/c10 reachable with `#include <c10/...>`
# - exported -DC10_USING_CUSTOM_GENERATED_MACROS compiler flag
# So, just patch our generated framework to do that.
sed -i '' '1i\
#define C10_USING_CUSTOM_GENERATED_MACROS
' \
"$HEADERS_PATH/executorch/runtime/core/portable_type/c10/c10/macros/Macros.h" \
"$HEADERS_PATH/executorch/runtime/core/portable_type/c10/c10/macros/Export.h"

cp -r $HEADERS_PATH/executorch/runtime/core/portable_type/c10/c10 "$HEADERS_PATH/"

cp "$SOURCE_ROOT_DIR/extension/apple/ExecuTorch/Exported/"*.h "$HEADERS_PATH/executorch"
cp "$SOURCE_ROOT_DIR/extension/apple/ExecuTorch/Exported/"*.modulemap "$HEADERS_PATH"

echo "Creating frameworks"

append_framework_flag() {
  local flag="$1"
  local framework="$2"
  local mode="${3:-}"
  if [[ $flag == ON ]]; then
    if [[ -n "$mode" && "$mode" != "Release" ]]; then
      local name spec
      name=$(echo "$framework" | cut -d: -f1)
      spec=$(echo "$framework" | cut -d: -f2-)
      framework="${name}_$(echo "$mode" | tr '[:upper:]' '[:lower:]'):${spec}"
    fi
    echo "Framework: $framework"
    FRAMEWORK_FLAGS+=("--framework=$framework")
  fi
}

for mode in "${MODES[@]}"; do
  FRAMEWORK_FLAGS=()
  for platform in "${PLATFORMS[@]}"; do
    echo "Directory: $platform/$mode"
    FRAMEWORK_FLAGS+=("--directory=$platform/$mode")
  done

  append_framework_flag "ON" "$FRAMEWORK_EXECUTORCH" "$mode"
  append_framework_flag "$COREML" "$FRAMEWORK_BACKEND_COREML" "$mode"
  append_framework_flag "$MPS" "$FRAMEWORK_BACKEND_MPS" "$mode"
  append_framework_flag "$XNNPACK" "$FRAMEWORK_BACKEND_XNNPACK" "$mode"
  append_framework_flag "$CUSTOM" "$FRAMEWORK_KERNELS_CUSTOM" "$mode"
  append_framework_flag "$OPTIMIZED" "$FRAMEWORK_KERNELS_OPTIMIZED" "$mode"
  append_framework_flag "$PORTABLE" "$FRAMEWORK_KERNELS_PORTABLE" "$mode"
  append_framework_flag "$QUANTIZED" "$FRAMEWORK_KERNELS_QUANTIZED" "$mode"

  "$SOURCE_ROOT_DIR"/build/create_frameworks.sh "${FRAMEWORK_FLAGS[@]}"
done

echo "Cleaning up"

for platform in "${PLATFORMS[@]}"; do
  rm -rf "$platform"
done

rm -rf "$HEADERS_PATH"

echo "Build succeeded!"
