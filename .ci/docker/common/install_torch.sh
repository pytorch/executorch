#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_torch() {
  # Clone the Executorch
  git clone https://github.com/pytorch/pytorch.git

  # Fetch the target commit
  pushd pytorch
  git checkout "${TORCH_VERSION}"
  git submodule update --init --recursive

  # Then build and install PyTorch
  conda_run python setup.py bdist_wheel
  pip_install "$(echo dist/*.whl)"
  popd
}

install_domains() {
  echo "Install torchvision and torchaudio"
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/audio.git@${TORCHAUDIO_VERSION}"
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/vision.git@${TORCHVISION_VERSION}"
}

install_torch
install_domains
