# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

MODEL_PATH=/data/local/tmp/llama3/llama3.pte
WEIGHT_SHARED_MODEL_PATH=/data/local/tmp/llama3/weights.ptd
POS_EMBEDDING_PATH=/data/local/tmp/llama3/pos_embedding.bin
TOKENIZER_PATH=/data/local/tmp/llama3/tokenizer.model
CONFIG_PATH="/data/local/tmp/exynos_llama3_1b.json"

SEQ_LEN=1024
TEMPERATURE=0
NUM_BOS=1
INPUT_PROMPT="What is the capital of France?"

chmod +x enn_executor_llm_runner

./enn_executor_llm_runner \
    --model_path=$MODEL_PATH \
    --data_path=$WEIGHT_SHARED_MODEL_PATH \
    --pos_embedding_path=$POS_EMBEDDING_PATH \
    --tokenizer_path=$TOKENIZER_PATH \
    --config_path "$CONFIG_PATH" \
    --prompt="$INPUT_PROMPT" \
    --temperature=$TEMPERATURE \
    --seq_len=$SEQ_LEN \
    --num_bos=$NUM_BOS \
    
