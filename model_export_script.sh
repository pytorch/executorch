set -e

export MODEL_IN=$HOME/models/stories110M/stories110M.pt
export TOKENIZER=$HOME/models/stories110M/tokenizer.bin
export PARAMS=$HOME/models/stories110M/params.json
export MODEL_OUT_DIR=$HOME/models/stories110M

export STATIC_SEQ_LENGTH=500

export MODEL_OUT_DECODE=${MODEL_OUT_DIR}/decode_model_${STATIC_SEQ_LENGTH}.pte

python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_DECODE -E "4,32" -kv --use_sdpa_with_kv_cache --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_and_ne --max_seq_length 1024 --verbose -d "fp16" --static_seq_length $STATIC_SEQ_LENGTH
python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_DECODE -o "${MODEL_OUT_DIR}/decode_${STATIC_SEQ_LENGTH}"
