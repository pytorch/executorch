#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# A prebuilt index can list .deb versions the archive has already dropped, and
# the install then fails on a 404 rather than on anything about this package.
# Guarded so a machine without apt still gets everything below, as the sibling
# scripts in this repository do.
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ffmpeg
fi
pip install torchcodec==0.11.0 --extra-index-url https://download.pytorch.org/whl/test/cpu
pip install moshi==0.2.11
pip install bitsandbytes soundfile einops
# Run llama2/install requirements for torchao deps
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
bash -e "$SCRIPT_DIR"/../../llama/install_requirements.sh
