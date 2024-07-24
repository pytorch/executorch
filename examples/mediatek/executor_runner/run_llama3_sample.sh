 # Copyright (c) 2024 MediaTek Inc.
 #
 # Licensed under the BSD License (the "License"); you may not use this file
 # except in compliance with the License. See the license file in the root
 # directory of this source tree for more details.

# Runtime
MAX_RESPONSE=200

# Model External
PROMPT_TOKEN_BATCH_SIZE=128
CACHE_SIZE=512

# Model Internals
HIDDEN_SIZE=4096
NUM_HEAD=32
NUM_LAYER=32
MAX_TOKEN_LENGTH=8192
ROT_EMB_BASE=500000

# Model IO Types
INPUT_TYPE=fp32
OUTPUT_TYPE=fp32
CACHE_TYPE=fp32
MASK_TYPE=fp32
ROT_EMB_TYPE=fp32

# Tokenizer
VOCAB_SIZE=128000
BOS_TOKEN=128000
EOS_TOKEN=128001
TOKENIZER_TYPE=tiktoken  # Use "bpe" for LLAMA2, "tiktoken" for LLAMA3

# Paths
TOKENIZER_PATH="/data/local/tmp/llama3/tokenizer.model"
TOKEN_EMBEDDING_PATH="/data/local/tmp/llama3/embedding_llama3_8b_instruct_fp32.bin"

# Comma-Separated Paths
PROMPT_MODEL_PATHS="\
/data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_128t512c_0.pte,\
/data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_128t512c_1.pte,\
/data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_128t512c_2.pte,\
/data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_128t512c_3.pte,"

# Comma-Separated Paths
GEN_MODEL_PATHS="\
/data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_1t512c_0.pte,\
/data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_1t512c_1.pte,\
/data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_1t512c_2.pte,\
/data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_1t512c_3.pte,"

PROMPT_FILE=/data/local/tmp/llama3/sample_prompt.txt

chmod +x mtk_llama_executor_runner

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD

./mtk_llama_executor_runner \
    --max_response=$MAX_RESPONSE \
    --prompt_token_batch_size=$PROMPT_TOKEN_BATCH_SIZE \
    --cache_size=$CACHE_SIZE \
    --hidden_size=$HIDDEN_SIZE \
    --num_head=$NUM_HEAD \
    --num_layer=$NUM_LAYER \
    --max_token_length=$MAX_TOKEN_LENGTH \
    --rot_emb_base=$ROT_EMB_BASE \
    --input_type=$INPUT_TYPE \
    --output_type=$OUTPUT_TYPE \
    --cache_type=$CACHE_TYPE \
    --mask_type=$MASK_TYPE \
    --rot_emb_type=$ROT_EMB_TYPE \
    --vocab_size=$VOCAB_SIZE \
    --bos_token=$BOS_TOKEN \
    --eos_token=$EOS_TOKEN \
    --tokenizer_type=$TOKENIZER_TYPE \
    --tokenizer_path=$TOKENIZER_PATH \
    --token_embedding_path=$TOKEN_EMBEDDING_PATH \
    --prompt_model_paths=$PROMPT_MODEL_PATHS \
    --gen_model_paths=$GEN_MODEL_PATHS \
    --prompt_file=$PROMPT_FILE