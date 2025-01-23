#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Download and prepare stories llama model artifacts
prepare_model_artifacts() {
    echo "Preparing stories model artifacts"
    wget -O stories110M.pt "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
    wget -O tokenizer.model "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"
    echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
}

prepare_model_artifacts
