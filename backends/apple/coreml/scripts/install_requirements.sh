#!/usr/bin/env bash
#
# Copyright © 2023 Apple Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR_PATH="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

# TODO(jathu): remove the need to fetch coremltools to build deps for coreml_executor_runner.
# Keep this version in sync with: pyproject.toml
COREMLTOOLS_VERSION="9.0"

# Safe colors (no TERM noise in CI)
if command -v tput >/dev/null 2>&1 && [ -t 1 ] && [ -n "${TERM:-}" ]; then
    red="$(tput setaf 1)"
    green="$(tput setaf 2)"
    reset="$(tput sgr0)"
else
    red=""; green=""; reset=""
fi

EXECUTORCH_ROOT_PATH=$(realpath "$SCRIPT_DIR_PATH/../../../../")
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/apple/coreml"
COREMLTOOLS_DIR_PATH="$COREML_DIR_PATH/third-party/coremltools"
PROTOBUF_FILES_DIR_PATH="$COREMLTOOLS_DIR_PATH/build/mlmodel/format/"

cd "$EXECUTORCH_ROOT_PATH"

rm -rf "$COREML_DIR_PATH/third-party"
mkdir -p "$COREML_DIR_PATH/third-party"

echo "${green}ExecuTorch: Cloning coremltools.${reset}"
git clone --depth 1 --branch "${COREMLTOOLS_VERSION}" "https://github.com/apple/coremltools.git" "$COREMLTOOLS_DIR_PATH"
cd "$COREMLTOOLS_DIR_PATH"

STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to clone coremltools.${reset}"
    exit 1
fi

# ---------------------------------------------------------------------
# Host toolchain / SDK setup JUST for coremltools build
# ---------------------------------------------------------------------
HOST_SDKROOT="${SDKROOT:-}"
HOST_CC="${CC:-}"
HOST_CXX="${CXX:-}"
HOST_CFLAGS="${CFLAGS:-}"
HOST_CXXFLAGS="${CXXFLAGS:-}"

if [[ "$(uname)" == "Darwin" ]]; then
    # Only pick macOS SDK if nothing else is specified
    if [[ -z "$HOST_SDKROOT" ]]; then
        HOST_SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
    fi
    if [[ -z "$HOST_CC" ]]; then
        HOST_CC="$(xcrun --find clang)"
    fi
    if [[ -z "$HOST_CXX" ]]; then
        HOST_CXX="$(xcrun --find clang++)"
    fi
    # Only add -isysroot if caller didn't already set CFLAGS/CXXFLAGS
    if [[ -z "$HOST_CFLAGS" && -n "$HOST_SDKROOT" ]]; then
        HOST_CFLAGS="-isysroot ${HOST_SDKROOT}"
    fi
    if [[ -z "$HOST_CXXFLAGS" && -n "$HOST_SDKROOT" ]]; then
        HOST_CXXFLAGS="-isysroot ${HOST_SDKROOT}"
    fi
fi

echo "${green}ExecuTorch: Installing coremltools dependencies.${reset}"
SDKROOT="$HOST_SDKROOT" \
CC="$HOST_CC" \
CXX="$HOST_CXX" \
CFLAGS="$HOST_CFLAGS" \
CXXFLAGS="$HOST_CXXFLAGS" \
python -m pip install -r "$COREMLTOOLS_DIR_PATH/reqs/build.pip"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to install coremltools dependencies.${reset}"
    exit 1
fi

mkdir -p "$COREMLTOOLS_DIR_PATH/build"

echo "${green}ExecuTorch: Configuring coremltools CMake build.${reset}"
SDKROOT="$HOST_SDKROOT" \
CC="$HOST_CC" \
CXX="$HOST_CXX" \
CFLAGS="$HOST_CFLAGS" \
CXXFLAGS="$HOST_CXXFLAGS" \
cmake -S "$COREMLTOOLS_DIR_PATH" -B "$COREMLTOOLS_DIR_PATH/build"

echo "${green}ExecuTorch: Building mlmodel target.${reset}"
SDKROOT="$HOST_SDKROOT" \
CC="$HOST_CC" \
CXX="$HOST_CXX" \
CFLAGS="$HOST_CFLAGS" \
CXXFLAGS="$HOST_CXXFLAGS" \
cmake --build "$COREMLTOOLS_DIR_PATH/build" --parallel --target mlmodel

echo "${green}ExecuTorch: Copying protobuf files.${reset}"
rm -rf "$COREML_DIR_PATH/runtime/sdk/format/"
mkdir -p "$COREML_DIR_PATH/runtime/sdk/format/"
cp -rf "$PROTOBUF_FILES_DIR_PATH" "$COREML_DIR_PATH/runtime/sdk/format/"
