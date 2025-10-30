#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -x

conda install -c conda-forge "ffmpeg<8" -y
pip install torchcodec==0.7.0.dev20251012 --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install moshi==0.2.11
pip install bitsandbytes soundfile einops
# Run llama2/install requirements for torchao deps
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
bash "$SCRIPT_DIR"/../../llama/install_requirements.sh
