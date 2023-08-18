#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

install_buck() {
  if ! command -v zstd &> /dev/null; then
    brew install zstd
  fi

  if ! command -v wget &> /dev/null; then
    brew install wget
  fi

  if ! command -v buck2 &> /dev/null; then
    pushd .ci/docker

    BUCK2=buck2-aarch64-apple-darwin.zst
    BUCK2_VERSION=$(cat ci_commit_pins/buck2.txt)

    wget -q "https://github.com/facebook/buck2/releases/download/${BUCK2_VERSION}/${BUCK2}"
    zstd -d "${BUCK2}" -o buck2

    chmod +x buck2
    mv buck2 /opt/homebrew/bin

    rm "${BUCK2}"
    popd
  fi
}

install_conda() {
  pushd .ci/docker
  # Install conda dependencies like flatbuffer
  conda install --file conda-env-ci.txt
  popd
}

install_pip_dependencies() {
  pushd .ci/docker
  # Install all Python dependencies, including PyTorch
  pip install --progress-bar off -r requirements-ci.txt

  TORCH_VERSION=$(cat ci_commit_pins/pytorch.txt)
  TORCHVISION_VERSION=$(cat ci_commit_pins/vision.txt)
  pip install --progress-bar off --pre torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/nightly/cpu
  popd
}

install_buck
install_conda
install_pip_dependencies
