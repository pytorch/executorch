#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

IMAGE_NAME="$1"
shift

echo "Building ${IMAGE_NAME} Docker image"

OS=ubuntu
OS_VERSION=22.04
CLANG_VERSION=""
GCC_VERSION=""
PYTHON_VERSION=3.10
MINICONDA_VERSION=23.10.0-1
BUCK2_VERSION=$(cat ci_commit_pins/buck2.txt)

case "${IMAGE_NAME}" in
  executorch-ubuntu-22.04-gcc9)
    LINTRUNNER=""
    GCC_VERSION=9
    ;;
  executorch-ubuntu-22.04-clang12)
    LINTRUNNER=""
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-linter)
    LINTRUNNER=yes
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-arm-sdk)
    ARM_SDK=yes
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-clang12-android)
    LINTRUNNER=""
    CLANG_VERSION=12
    # From https://developer.android.com/ndk/downloads
    ANDROID_NDK_VERSION=r26c
    ;;
  *)
    echo "Invalid image name ${IMAGE_NAME}"
    exit 1
esac

TORCH_VERSION=$(cat ci_commit_pins/pytorch.txt)
BUILD_DOCS=1

# Copy requirements-lintrunner.txt from root to here
cp ../../requirements-lintrunner.txt ./

# Copy arm setup script from root to here
# TODO(huydhn): Figure out a way to rebuild the Docker image automatically
# with a new image hash when the content here is updated
cp -r ../../examples/arm/ ./arm

docker build \
  --no-cache \
  --progress=plain \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  --build-arg "GCC_VERSION=${GCC_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "MINICONDA_VERSION=${MINICONDA_VERSION}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "BUCK2_VERSION=${BUCK2_VERSION}" \
  --build-arg "LINTRUNNER=${LINTRUNNER:-}" \
  --build-arg "BUILD_DOCS=${BUILD_DOCS}" \
  --build-arg "ARM_SDK=${ARM_SDK:-}" \
  --build-arg "ANDROID_NDK_VERSION=${ANDROID_NDK_VERSION:-}" \
  -f "${OS}"/Dockerfile \
  "$@" \
  .
