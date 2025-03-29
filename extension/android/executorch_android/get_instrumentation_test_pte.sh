#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

mkdir -p "${SCRIPT_DIR}/src/androidTest/resources"
cp "${SCRIPT_DIR}/../../../extension/module/test/resources/add.pte" "${SCRIPT_DIR}/src/androidTest/resources"

pushd "${SCRIPT_DIR}/../../../"
curl -Ls "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt" --output stories110M.pt
curl -Ls "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model" --output tokenizer.model
touch params.json
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
python -m examples.models.llama.export_llama -X --xnnpack-extended-ops -qmode 8da4w -G 128 -c stories110M.pt -p params.json --output_name tinyllama_portable_fp16_h.pte
mv tinyllama_portable_fp16_h.pte "${SCRIPT_DIR}/src/androidTest/resources"
rm stories110M.pt tokenizer.model params.json
popd
