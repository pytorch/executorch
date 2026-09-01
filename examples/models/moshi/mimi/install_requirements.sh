#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# The image's apt index is stale, so refresh it before installing: the .deb
# versions it lists may already be gone from the archive.
sudo apt-get update
sudo apt-get install -y ffmpeg
pip install torchcodec==0.11.0 --extra-index-url https://download.pytorch.org/whl/test/cpu
pip install moshi==0.2.11
pip install bitsandbytes soundfile einops
# Run llama2/install requirements for torchao deps
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
bash "$SCRIPT_DIR"/../../llama/install_requirements.sh
