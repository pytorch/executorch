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
HIDDEN_SIZE=3584
NUM_HEAD=28
NUM_LAYER=28
MAX_TOKEN_LENGTH=32768
ROT_EMB_BASE=1000000

# Model IO Types
INPUT_TYPE=fp32
OUTPUT_TYPE=fp32
CACHE_TYPE=fp32
MASK_TYPE=fp32
ROT_EMB_TYPE=fp32

# Tokenizer
VOCAB_SIZE=152064
BOS_TOKEN=151643
EOS_TOKEN=151645
TOKENIZER_TYPE=hf  # Use "bpe" for LLAMA2, "tiktoken" for LLAMA3, "hf" for huggingface tokenizer

# Paths
TOKENIZER_PATH="/data/local/tmp/et_mtk/tokenizer_qwen3.json"
TOKEN_EMBEDDING_PATH="/data/local/tmp/et_mtk/embedding_Qwen2-7B-Instruct_fp32.bin"

# Comma-Separated Paths
WEIGHT_SHARED_MODEL_PACKAGE_PATHS="\
/data/local/tmp/et_mtk/Qwen2-7B-Instruct_A16W4_4_chunks/Qwen2-7B-Instruct_A16W4_4_chunks_0.pte,\
/data/local/tmp/et_mtk/Qwen2-7B-Instruct_A16W4_4_chunks/Qwen2-7B-Instruct_A16W4_4_chunks_1.pte,\
/data/local/tmp/et_mtk/Qwen2-7B-Instruct_A16W4_4_chunks/Qwen2-7B-Instruct_A16W4_4_chunks_2.pte,\
/data/local/tmp/et_mtk/Qwen2-7B-Instruct_A16W4_4_chunks/Qwen2-7B-Instruct_A16W4_4_chunks_3.pte,"

PROMPT_FILE=/data/local/tmp/et_mtk/prompt.txt

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
    --model_package_paths=$WEIGHT_SHARED_MODEL_PACKAGE_PATHS \
    --prompt_file=$PROMPT_FILE