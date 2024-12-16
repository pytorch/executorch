set -e

export MODEL_IN=$HOME/models/stories110M/stories110M.pt
export TOKENIZER=$HOME/models/stories110M/tokenizer.bin
export PARAMS=$HOME/models/stories110M/params.json
export MODEL_OUT_DIR=$HOME/models/stories110M
export MODEL_OUT_PREFILL=$MODEL_OUT_DIR/prefill_model.pte
export MODEL_OUT_DECODE=$MODEL_OUT_DIR/decode_model.pte
export MODEL_OUT_DECODE_KV_IO=$MODEL_OUT_DIR/decode_kv_io_model.pte
export MODEL_OUT_DECODE_KV_IO_ADDITIVE=$MODEL_OUT_DIR/decode_kv_io_additive_model.pte


python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_PREFILL -E "4,32" --prefill_seq_length 512 --disable_dynamic_shape --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_and_ne --max_seq_length 1024 --prefill_return_kv --dtype fp16
python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_DECODE -E "4,32" -kv --disable_dynamic_shape --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_and_ne --max_seq_length 1024
python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_DECODE_KV_IO -E "4,32" -kv --disable_dynamic_shape --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_and_ne --max_seq_length 1024 --decode_kv_cache_as_io --dtype fp16
python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_DECODE_KV_IO_ADDITIVE -E "4,32" -kv --disable_dynamic_shape --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_and_ne --max_seq_length 1024 --decode_kv_cache_as_io --use_additive_kv_cache_update --dtype fp16


python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_PREFILL -o "${MODEL_OUT_DIR}/prefill"
python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_DECODE -o "${MODEL_OUT_DIR}/decode"
python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_DECODE_KV_IO -o "${MODEL_OUT_DIR}/decode_kv_io"
python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_DECODE_KV_IO_ADDITIVE -o "${MODEL_OUT_DIR}/decode_kv_io_additive"

python combine_coreml_models.py -m1 "${MODEL_OUT_DIR}/prefill/extracted_coreml_models/model_1/lowered_module/model.mlpackage" -m2 "${MODEL_OUT_DIR}/decode/extracted_coreml_models/model_1/lowered_module/model.mlpackage" -o "${MODEL_OUT_DIR}/combined.mlpackage"
