#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# torch_pin lives at the executorch repo root.
cd "$SCRIPT_DIR/../../../.."

TORCHCODEC_PKG=$(python -c "from torch_pin import torchcodec_spec; print(torchcodec_spec())")
TORCHCODEC_INDEX=$(python -c "from torch_pin import torch_index_url_base; print(torch_index_url_base())")

sudo apt install ffmpeg -y
pip install "$TORCHCODEC_PKG" --extra-index-url "${TORCHCODEC_INDEX}/cpu"
pip install moshi==0.2.11
pip install bitsandbytes soundfile einops
# Run llama2/install requirements for torchao deps
bash "$SCRIPT_DIR"/../../llama/install_requirements.sh
