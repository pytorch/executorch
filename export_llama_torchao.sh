# No quantization
# Set these paths to point to the downloaded files
LLAMA_CHECKPOINT=../../../.llama/checkpoints/Llama-3.2-3B-Instruct/original/consolidated.00.pth
LLAMA_PARAMS=../../../.llama/checkpoints/Llama-3.2-3B-Instruct/original/params.json

# Set low-bit quantization parameters
QLINEAR_BITWIDTH=4       # Can be 1-8
QLINEAR_GROUP_SIZE=128   # Must be multiple of 16
QEMBEDDING_BITWIDTH=4    # Can be 1-8
QEMBEDDING_GROUP_SIZE=32 # Must be multiple of 16

python -m extension.llm.export.export_llm \
  base.model_class="llama3_2" \
  base.checkpoint="${LLAMA_CHECKPOINT:?}" \
  base.params="${LLAMA_PARAMS:?}" \
  model.use_kv_cache=True \
  model.use_sdpa_with_kv_cache=True \
  base.metadata='"{\"get_bos_id\":128000, \"get_eos_ids\":[128009, 128001]}"' \
  export.output_name="llama3_2.pte" \
  quantization.qmode="torchao:8da${QLINEAR_BITWIDTH}w" \
  quantization.group_size=${QLINEAR_GROUP_SIZE} \
  quantization.embedding_quantize=\'torchao:${QEMBEDDING_BITWIDTH},${QEMBEDDING_GROUP_SIZE}\' \
  model.dtype_override="fp32"
