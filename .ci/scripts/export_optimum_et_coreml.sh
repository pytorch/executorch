#!/bin/bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

MODEL_NAME="$1"
OUTPUT_DIR="$2"
MODEL_ID=""
TASK=""
RECIPE=""

# Map model name to model_id
case "$MODEL_NAME" in
  smollm)
    MODEL_ID="HuggingFaceTB/SmolLM2-135M"
    TASK="text-generation"
    RECIPE="coreml_llm_4bit"
    ;;
  llama3)
    MODEL_ID="NousResearch/Llama-3.2-1B"
    TASK="text-generation"
    RECIPE="coreml_llm_4bit"
    ;;
  *)
    echo "Error: Unknown model name '$MODEL_NAME'"
    exit 1
    ;;
esac



# Call the CLI tool with the resolved model_id
echo "Exporting model: $MODEL_NAME (ID: $MODEL_ID, TASK: $TASK, RECIPE: $RECIPE)"
optimum-cli export executorch \
                    --model "${MODEL_ID}" \
                    --task "${TASK}" \
                    --recipe "${RECIPE}" \
                    --output_dir ${OUTPUT_DIR}
