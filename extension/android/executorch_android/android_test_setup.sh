#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi
which "${PYTHON_EXECUTABLE}"

BASEDIR=$(dirname "$0")
cp "${BASEDIR}/../../../extension/module/test/resources/add.pte" "${BASEDIR}/src/androidTest/resources"

pushd "${BASEDIR}/../../../"
curl -Ls "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt" --output stories110M.pt
curl -Ls "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model" --output tokenizer.model
# Create params.json file
touch params.json
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
python -m examples.models.llama.export_llama -c stories110M.pt -p params.json -X -kv --model=stories110m

cp *.pte "${BASEDIR}/src/androidTest/resources/stories.pte"
cp tokenizer.model "${BASEDIR}/src/androidTest/resources/tokenizer.bin"
popd
