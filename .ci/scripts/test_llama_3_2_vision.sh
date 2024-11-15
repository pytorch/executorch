#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

ENABLE_KV_CACHE="${1:-false}"

if [[ "${ENABLE_KV_CACHE}" != "true" && "${ENABLE_KV_CACHE}" != "false" ]]; then
    echo "Error: ENABLE_KV_CACHE must be 'true' or 'false'"
    exit 1
fi

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
    PYTHON_EXECUTABLE=python3
fi

download_dependencies() {
    bash examples/models/llama3_2_vision/install_requirements.sh
    tune download meta-llama/Llama-3.2-11B-Vision-Instruct --output-dir /tmp/Llama-3.2-11B-Vision-Instruct
}

run_and_verify_eager() {
    NOW=$(date +"%H:%M:%S")
    echo "Starting to test llama3_2_vision text decoder at ${NOW}"
    if [[ ! -f "/tmp/Llama-3.2-11B-Vision-Instruct/original/consolidated.pth" ]]; then
        echo "checkpoint (consolidated.pth) is missing."
        exit 1
    fi
    if [[ ! -f "/tmp/Llama-3.2-11B-Vision-Instruct/original/tokenizer.model" ]]; then
        echo "tokenizer.model is missing."
        exit 1
    fi
    
    EAGER_RUNNER_ARGS="$PYTHON_EXECUTABLE -m examples.models.llama3_2_vision.runner.eager \
	-c /tmp/Llama-3.2-11B-Vision-Instruct/original/consolidated.pth \
	-t /tmp/Llama-3.2-11B-Vision-Instruct/original/tokenizer.model \
	-d fp32 \
	--max_seq_length 32 \
	--temperature 0 \
        --show_tokens \
	--prompt \"Once upon a time,\" > result.txt"

    if [[ "${ENABLE_KV_CACHE}" == "true" ]]; then
	EAGER_RUNNER_ARGS="${EAGER_RUNNER_ARGS} -kv"
    fi

    # Verify result.txt
    RESULT=$(cat result.txt)
    EXPECTED_RESULT="727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 18012, 304, 1708, 5377, 297, 278, 6575, 845, 457, 29889, 3118, 2462, 29892, 1183, 4446, 263"
    if [[ "${RESULT}" == *"${EXPECTED_RESULT}"* ]]; then
        echo "Actual result: ${RESULT}"
        echo "Success"
        exit 0
    else
        echo "Actual result: ${RESULT}"
        echo "Failure; results not the same"
        exit 1
    fi
}

download_dependencies
run_and_verify
