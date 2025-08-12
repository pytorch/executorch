#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

BASEDIR=$(dirname "$0")
pushd "$BASEDIR"/../../../../
curl -C - -Ls "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt" --output stories110M.pt
curl -C - -Ls "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model" --output tokenizer.model
# Create params.json file
touch params.json
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
python -m extension.llm.export.export_llm base.checkpoint=stories110M.pt base.params=params.json model.dtype_override=fp16 export.output_name=stories110m_h.pte model.use_kv_cache=true
python -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin

adb mkdir -p /data/local/tmp/llama
adb push stories110m_h.pte /data/local/tmp/llama
adb push tokenizer.bin /data/local/tmp/llama
popd

pushd "$BASEDIR"
./gradlew connectedAndroidTest
popd
