#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Download prebuilt wgpu-native binaries for native (non-browser) WebGPU testing.
# Usage: bash backends/webgpu/scripts/setup-wgpu-native.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WGPU_DIR="${SCRIPT_DIR}/../third-party/wgpu-native"

WGPU_VERSION="v27.0.4.0"
WGPU_BASE_URL="https://github.com/gfx-rs/wgpu-native/releases/download/${WGPU_VERSION}"

if [[ -f "${WGPU_DIR}/lib/libwgpu_native.a" ]]; then
    echo "wgpu-native already installed at ${WGPU_DIR}"
    exit 0
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)
        echo "Unsupported OS: ${OS}"
        exit 1
        ;;
esac

case "${ARCH}" in
    x86_64)  WGPU_ARCH="x86_64" ;;
    aarch64|arm64) WGPU_ARCH="aarch64" ;;
    *)
        echo "Unsupported architecture: ${ARCH}"
        exit 1
        ;;
esac

ZIP_NAME="wgpu-${PLATFORM}-${WGPU_ARCH}-release.zip"
URL="${WGPU_BASE_URL}/${ZIP_NAME}"

echo "Downloading wgpu-native ${WGPU_VERSION} for ${PLATFORM}-${WGPU_ARCH}..."
TMPDIR_DL="$(mktemp -d)"
trap "rm -rf ${TMPDIR_DL}" EXIT

curl -sL "${URL}" -o "${TMPDIR_DL}/${ZIP_NAME}"

mkdir -p "${WGPU_DIR}"
unzip -qo "${TMPDIR_DL}/${ZIP_NAME}" -d "${WGPU_DIR}"

echo "Installed wgpu-native to ${WGPU_DIR}"
ls -la "${WGPU_DIR}/lib/"
