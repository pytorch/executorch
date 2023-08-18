#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

install_executorch() {
  which pip
  # Install executorch, this assumes that Executorch is checked out in the
  # current directory
  pip install .
  # Just print out the list of packages for debugging
  pip list
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
  TORCHAUDIO_VERSION=$(cat ci_commit_pins/audio.txt)
  TORCHVISION_VERSION=$(cat ci_commit_pins/vision.txt)
  pip install --progress-bar off --pre \
    torch=="${TORCH_VERSION}" \
    torchaudio=="${TORCHAUDIO_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    --index-url https://download.pytorch.org/whl/nightly/cpu
  popd
}
