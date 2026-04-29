#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

FULL_IMAGE_NAME="$1"
shift

IMAGE_NAME=$(echo "${FULL_IMAGE_NAME}" | sed 's/ci-image://')

echo "Building ${IMAGE_NAME} Docker image"

OS=ubuntu
OS_VERSION=22.04
CLANG_VERSION=""
GCC_VERSION=""
PYTHON_VERSION=3.10
MINICONDA_VERSION=23.10.0-1
BUCK2_VERSION=$(cat ci_commit_pins/buck2.txt)

case "${IMAGE_NAME}" in
  executorch-ubuntu-22.04-gcc11)
    LINTRUNNER=""
    GCC_VERSION=11
    ;;
  executorch-ubuntu-22.04-gcc9-nopytorch)
    LINTRUNNER=""
    GCC_VERSION=9
    SKIP_PYTORCH=yes
    ;;
  executorch-ubuntu-22.04-clang12)
    LINTRUNNER=""
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-gcc11-aarch64)
    LINTRUNNER=""
    GCC_VERSION=11
    ;;
  executorch-ubuntu-22.04-gcc11-aarch64-android)
    LINTRUNNER=""
    GCC_VERSION=11
    ANDROID_NDK_VERSION=r28c
    ;;
  executorch-ubuntu-22.04-gcc11-aarch64-arm-sdk)
    ARM_SDK=yes
    GCC_VERSION=11
    ;;
  executorch-ubuntu-22.04-linter)
    LINTRUNNER=yes
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-arm-sdk)
    ARM_SDK=yes
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-zephyr-sdk)
    ZEPHYR_SDK=yes
    GCC_VERSION=11
    ;;
  executorch-ubuntu-22.04-qnn-sdk)
    QNN_SDK=yes
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-mediatek-sdk)
    MEDIATEK_SDK=yes
    CLANG_VERSION=12
    ANDROID_NDK_VERSION=r28c
    ;;
  executorch-ubuntu-22.04-clang12-android)
    LINTRUNNER=""
    CLANG_VERSION=12
    # From https://developer.android.com/ndk/downloads
    ANDROID_NDK_VERSION=r28c
    ;;
  executorch-ubuntu-22.04-cuda-windows)
    LINTRUNNER=""
    GCC_VERSION=11
    CUDA_WINDOWS_CROSS_COMPILE=yes
    CUDA_VERSION=12.8
    SKIP_PYTORCH=yes
    ;;
  *)
    echo "Invalid image name ${IMAGE_NAME}"
    exit 1
esac

TORCH_VERSION=$(cat ci_commit_pins/pytorch.txt)
BUILD_DOCS=1

# Pull channel + spec/url helpers out of torch_pin.py so install_pytorch.sh
# (which runs inside the docker build, where torch_pin.py isn't available)
# can decide between wheel install (test/release) and source build (nightly).
# Self-hosted runners often have python3 but not the unversioned python alias.
PYTHON_BIN=$(command -v python3 || command -v python)
TORCH_PIN_HELPERS=$(cd ../.. && "$PYTHON_BIN" -c "from torch_pin import CHANNEL, torch_spec, torchaudio_spec, torchvision_spec, torch_index_url_base; print(CHANNEL); print(torch_spec()); print(torchaudio_spec()); print(torchvision_spec()); print(torch_index_url_base())")
TORCH_CHANNEL=$(echo "${TORCH_PIN_HELPERS}" | sed -n '1p')
TORCH_SPEC=$(echo "${TORCH_PIN_HELPERS}" | sed -n '2p')
TORCHAUDIO_SPEC=$(echo "${TORCH_PIN_HELPERS}" | sed -n '3p')
TORCHVISION_SPEC=$(echo "${TORCH_PIN_HELPERS}" | sed -n '4p')
TORCH_INDEX_URL=$(echo "${TORCH_PIN_HELPERS}" | sed -n '5p')

# Copy requirements-lintrunner.txt from root to here
cp ../../requirements-lintrunner.txt ./

docker build \
  --no-cache \
  --progress=plain \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  --build-arg "GCC_VERSION=${GCC_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "MINICONDA_VERSION=${MINICONDA_VERSION}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "TORCH_CHANNEL=${TORCH_CHANNEL}" \
  --build-arg "TORCH_SPEC=${TORCH_SPEC}" \
  --build-arg "TORCHAUDIO_SPEC=${TORCHAUDIO_SPEC}" \
  --build-arg "TORCHVISION_SPEC=${TORCHVISION_SPEC}" \
  --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
  --build-arg "BUCK2_VERSION=${BUCK2_VERSION}" \
  --build-arg "LINTRUNNER=${LINTRUNNER:-}" \
  --build-arg "BUILD_DOCS=${BUILD_DOCS}" \
  --build-arg "ARM_SDK=${ARM_SDK:-}" \
  --build-arg "ZEPHYR_SDK=${ZEPHYR_SDK:-}" \
  --build-arg "QNN_SDK=${QNN_SDK:-}" \
  --build-arg "MEDIATEK_SDK=${MEDIATEK_SDK:-}" \
  --build-arg "ANDROID_NDK_VERSION=${ANDROID_NDK_VERSION:-}" \
  --build-arg "SKIP_PYTORCH=${SKIP_PYTORCH:-}" \
  --build-arg "CUDA_WINDOWS_CROSS_COMPILE=${CUDA_WINDOWS_CROSS_COMPILE:-}" \
  --build-arg "CUDA_VERSION=${CUDA_VERSION:-}" \
  -f "${OS}"/Dockerfile \
  "$@" \
  .
