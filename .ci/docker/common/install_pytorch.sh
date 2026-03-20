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
  pip_install --force-reinstall --no-cache-dir torchvision==0.26.0 torchaudio==2.11.0 torchao==0.16.0 --index-url https://download.pytorch.org/whl/test/cpu
}

install_pytorch_and_domains() {
  pip_install --force-reinstall --no-cache-dir torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 torchao==0.16.0 --index-url https://download.pytorch.org/whl/test/cpu
}

install_pytorch_and_domains
