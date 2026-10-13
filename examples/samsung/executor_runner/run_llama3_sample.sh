# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

MODEL_PATH=/data/local/tmp/llama3/llama3.pte
WEIGHT_SHARED_MODEL_PATH=/data/local/tmp/llama3/weights.ptd
TOKENIZER_PATH=/data/local/tmp/llama3/tokenizer.model
POS_EMBEDDING_PATH=/data/local/tmp/llama3/pos_embedding.bin

SEQ_LEN=1024
TEMPERATURE=0
INPUT_PROMPT="What is the capital of France?"

chmod +x enn_executor_llama_runner

./enn_executor_llama_runner \
    --model_path=$MODEL_PATH \
    --data_path=$WEIGHT_SHARED_MODEL_PATH \
    --seq_len=$SEQ_LEN \
    --temperature=$TEMPERATURE \
    --num_bos=1 \
    --pos_embedding_path=$POS_EMBEDDING_PATH \
    --tokenizer_type="tiktoken" \
    --tokenizer_path=$TOKENIZER_PATH \
    --prompt="$INPUT_PROMPT"
