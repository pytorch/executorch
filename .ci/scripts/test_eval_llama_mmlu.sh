#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
    PYTHON_EXECUTABLE=python3
fi

# Download and prepare stories model artifacts
prepare_model_artifacts() {
    echo "Preparing stories model artifacts"
    wget -O stories110M.pt "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
    wget -O tokenizer.model "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"
    echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
}

run_and_verify() {
    NOW=$(date +"%H:%M:%S")
    echo "Starting to run eval_llama at ${NOW}"
    if [[ ! -f "stories110M.pt" ]]; then
        echo "stories110M.pt is missing."
        exit 1
    fi
    if [[ ! -f "tokenizer.model" ]]; then
        echo "tokenizer.model is missing."
        exit 1
    fi
    if [[ ! -f "params.json" ]]; then
        echo "params.json is missing."
        exit 1
    fi
    $PYTHON_EXECUTABLE -m examples.models.llama.eval_llama \
	-c stories110M.pt \
	-p params.json \
	-t tokenizer.model \
	-kv \
	-d fp32 \
	--tasks mmlu \
	-f 5 \
	--max_seq_length 2048 \
	--limit 5 > result.txt

    # Verify result.txt
    RESULT=$(cat result.txt)
    EXPECTED_TASK="mmlu"
    EXPECTED_RESULT="acc"
    if [[ "${RESULT}" == "${EXPECTED_TASK}: {"*"${EXPECTED_RESULT}"* ]]; then
        echo "Actual result: ${RESULT}"
        echo "Success"
        exit 0
    else
        echo "Actual result: ${RESULT}"
        echo "Failure; results not the same"
        exit 1
    fi
}

prepare_model_artifacts
run_and_verify
