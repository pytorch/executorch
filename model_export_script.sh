set -e

export MODEL_IN=$HOME/Documents/llama/stories_110M/stories110M.pt
export TOKENIZER=$HOME/Documents/llama/stories_110M/tokenizer.bin
export PARAMS=$HOME/Documents/llama/stories_110M/params.json
export MODEL_OUT_DIR=$HOME/Documents/llama/stories_110M
export MODEL_OUT_DECODE_KV_IO=$MODEL_OUT_DIR/decode_kv_io_model.pte
export MODEL_OUT_DECODE_KV_IO_ADDITIVE=$MODEL_OUT_DIR/decode_kv_io_additive_model.pte

export STATIC_SEQ_LENGTH=1

export MODEL_OUT_DECODE=${MODEL_OUT_DIR}/decode_model_${STATIC_SEQ_LENGTH}.pte

# python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_DECODE -E "4,32" -kv --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_and_ne --max_seq_length 1024 --verbose -d "fp16" --static_seq_length $STATIC_SEQ_LENGTH
# python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_DECODE_KV_IO -E "4,32" -kv --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_and_ne --max_seq_length 1024 --verbose -d "fp16" --static_seq_length $STATIC_SEQ_LENGTH --decode_kv_cache_as_io
python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_DECODE_KV_IO_ADDITIVE -E "4,32" -kv --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_and_ne --max_seq_length 1024 --verbose -d "fp16" --static_seq_length $STATIC_SEQ_LENGTH --decode_kv_cache_as_io --use_additive_kv_cache_update

# python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_DECODE -o "${MODEL_OUT_DIR}/decode_${STATIC_SEQ_LENGTH}"
# python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_DECODE_KV_IO -o "${MODEL_OUT_DIR}/decode_kv_io${STATIC_SEQ_LENGTH}"
python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_DECODE_KV_IO_ADDITIVE -o "${MODEL_OUT_DIR}/decode_kv_io_additive${STATIC_SEQ_LENGTH}"
