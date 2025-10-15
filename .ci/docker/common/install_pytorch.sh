#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_domains() {
  echo "Install torchvision and torchaudio"
  pip install torchvision==0.24.0 torchaudio==2.9.0
}

install_pytorch_and_domains() {
  pip_install torch==2.9.0 torchvision=0.24.0 torchaudio=2.9.0 torchao==0.14.0
}

install_pytorch_and_domains
