
LLAMA_CHECKPOINT=<model_directory>/consolidated.00.pth
LLAMA_PARAMS=<model_directory>/params.json
LLAMA_TOKENIZER=<model_directory>/tokenizer.model

python -m extension.llm.export.export_llm \
  --config llama3_2_ov_4wo_config.yaml \
  +base.model_class="llama3_2" \
  +base.checkpoint="${LLAMA_CHECKPOINT:?}" \
  +base.params="${LLAMA_PARAMS:?}" \
  +base.tokenizer_path="${LLAMA_TOKENIZER:?}" \
