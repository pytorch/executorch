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
MINICONDA_VERSION=23.10.0-1
BUCK2_VERSION=$(cat ci_commit_pins/buck2.txt)

case "${IMAGE_NAME}" in
  executorch-ubuntu-22.04-clang12)
    LINTRUNNER=""
    ;;
  executorch-ubuntu-22.04-linter)
    LINTRUNNER=yes
    ;;
  executorch-ubuntu-22.04-arm-sdk)
    ARM_SDK=yes
    ;;
  *)
    echo "Invalid image name ${IMAGE_NAME}"
    exit 1
esac

# NB: All files needed to build the Docker image needs to be in .ci/docker
# folder so that the image hash is updated correctly when they change. The
# good news is that links can be setup to refer to them from other locations

TORCH_VERSION=$(cat ci_commit_pins/pytorch.txt)
BUILD_DOCS=1

docker build \
  --no-cache \
  --progress=plain \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "MINICONDA_VERSION=${MINICONDA_VERSION}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "BUCK2_VERSION=${BUCK2_VERSION}" \
  --build-arg "LINTRUNNER=${LINTRUNNER:-}" \
  --build-arg "BUILD_DOCS=${BUILD_DOCS}" \
  --build-arg "ARM_SDK=${ARM_SDK:-}" \
  -f "${OS}"/Dockerfile \
  "$@" \
  .
