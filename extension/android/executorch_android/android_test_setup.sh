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

BASEDIR=$(dirname "$(realpath $0)")

prepare_add() {
  cp "${BASEDIR}/../../../extension/module/test/resources/add.pte" "${BASEDIR}/src/androidTest/resources"
}

prepare_tinyllama() {
  pushd "${BASEDIR}/../../../"
  curl -C - -Ls "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt" --output stories15M.pt
  curl -C - -Ls "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model" --output tokenizer.model
  # Create params.json file
  touch params.json
  echo '{"dim": 288, "multiple_of": 32, "n_heads": 6, "n_layers": 6, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
  python -m examples.models.llama.export_llama -c stories15M.pt -p params.json -d fp16 -n stories15m_h.pte -kv
  python -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin

  cp stories15m_h.pte "${BASEDIR}/src/androidTest/resources/stories.pte"
  cp tokenizer.bin "${BASEDIR}/src/androidTest/resources/tokenizer.bin"
  popd
}

prepare_vision() {
  pushd "${BASEDIR}/../../../"
  python3 -m examples.xnnpack.aot_compiler --model_name "mv2" --delegate
  python3 -m examples.xnnpack.aot_compiler --model_name "mv3" --delegate
  python3 -m examples.xnnpack.aot_compiler --model_name "resnet50" --quantize --delegate
  cp mv2*.pte mv3*.pte resnet50*.pte "${BASEDIR}/src/androidTest/resources/"
  popd
}

prepare_add
prepare_tinyllama
prepare_vision
