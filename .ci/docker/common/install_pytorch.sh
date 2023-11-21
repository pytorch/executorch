#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_pytorch() {
  # Clone the Executorch
  git clone https://github.com/pytorch/pytorch.git

  # Fetch the target commit
  pushd pytorch || true
  git checkout "${TORCH_VERSION}"
  git submodule update --init --recursive

  chown -R ci-user .

  # Then build and install PyTorch
  conda_run python setup.py bdist_wheel
  pip_install "$(echo dist/*.whl)"
  popd || true

  # Clean up the cloned PyTorch repo to reduce the Docker image size
  rm -rf pytorch

  # Print sccache stats for debugging
  as_ci_user sccache --show-stats
}

install_domains() {
  echo "Install torchvision and torchaudio"
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/audio.git@${TORCHAUDIO_VERSION}"
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/vision.git@${TORCHVISION_VERSION}"
}

install_pytorch
install_domains
