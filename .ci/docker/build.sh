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
CLANG_VERSION=12
PYTHON_VERSION=3.10
MINICONDA_VERSION=23.5.1-0
BUCK2_VERSION=$(cat ci_commit_pins/buck2.txt)

NIGHTLY=$(cat ci_commit_pins/nightly.txt)
TORCH_VERSION=$(cat ci_commit_pins/pytorch.txt)
TORCHAUDIO_VERSION=$(cat ci_commit_pins/audio.txt)
TORCHVISION_VERSION=$(cat ci_commit_pins/vision.txt)
TIMM_VERSION=$(cat ci_commit_pins/timm.txt)

docker build \
  --no-cache \
  --progress=plain \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "MINICONDA_VERSION=${MINICONDA_VERSION}" \
  --build-arg "TIMM_VERSION=${TIMM_VERSION}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}.${NIGHTLY}" \
  --build-arg "TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}.${NIGHTLY}" \
  --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}.${NIGHTLY}" \
  --build-arg "BUCK2_VERSION=${BUCK2_VERSION}" \
  -f "${OS}"/Dockerfile \
  "$@" \
  .
