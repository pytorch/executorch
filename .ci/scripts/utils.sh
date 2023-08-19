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
  pushd .ci/docker || return
  # Install conda dependencies like flatbuffer
  conda install --file conda-env-ci.txt
  popd || return
}

install_pip_dependencies() {
  pushd .ci/docker || return
  # Install all Python dependencies, including PyTorch
  pip install --progress-bar off -r requirements-ci.txt

  NIGHTLY=$(cat ci_commit_pins/nightly.txt)
  TORCH_VERSION=$(cat ci_commit_pins/pytorch.txt)
  TORCHAUDIO_VERSION=$(cat ci_commit_pins/audio.txt)
  TORCHVISION_VERSION=$(cat ci_commit_pins/vision.txt)

  pip install --progress-bar off --pre \
    torch=="${TORCH_VERSION}.${NIGHTLY}" \
    torchaudio=="${TORCHAUDIO_VERSION}.${NIGHTLY}" \
    torchvision=="${TORCHVISION_VERSION}.${NIGHTLY}" \
    --index-url https://download.pytorch.org/whl/nightly/cpu
  popd || return
}
