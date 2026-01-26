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
  pip_install --force-reinstall torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
}

install_pytorch_and_domains() {
  pip_install --force-reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchao==0.15.0 --index-url https://download.pytorch.org/whl/cpu
}

install_pytorch_and_domains
