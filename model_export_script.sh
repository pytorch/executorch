set -e

export MODEL_IN=$HOME/models/stories110M/stories110M.pt
export TOKENIZER=$HOME/models/stories110M/tokenizer.bin
export PARAMS=$HOME/models/stories110M/params.json
export MODEL_OUT_DIR=$HOME/models/stories110M
export MODEL_OUT_PREFILL=$MODEL_OUT_DIR/prefill_model.pte
export MODEL_OUT_DECODE=$MODEL_OUT_DIR/decode_model.pte

python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_PREFILL -E "4,32" --prefill_seq_length 512 --disable_dynamic_shape --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_only --max_seq_length 1024 --prefill_return_kv --dtype fp16

python -m examples.models.llama.export_llama -c $MODEL_IN -p $PARAMS --output_name=$MODEL_OUT_DECODE -E "4,32" -kv --disable_dynamic_shape --coreml --coreml-ios 18 --coreml-quantize c4w --coreml-compute-units cpu_only --max_seq_length 1024


python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_PREFILL -o "${MODEL_OUT_DIR}/prefill"
python examples/apple/coreml/scripts/extract_coreml_models.py -m $MODEL_OUT_DECODE -o "${MODEL_OUT_DIR}/decode"

python combine_coreml_models.py -m1 "${MODEL_OUT_DIR}/prefill/extracted_coreml_models/model_1/lowered_module/model.mlpackage" -m2 "${MODEL_OUT_DIR}/decode/extracted_coreml_models/model_1/lowered_module/model.mlpackage" -o "${MODEL_OUT_DIR}/combined.mlpackage"
