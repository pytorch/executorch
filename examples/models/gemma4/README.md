# Summary

This example adds native ExecuTorch text-only export support for Google's Gemma 4 `E2B` and `E4B` models through the existing Llama-style export path.

The current scope is the decoder-only text model. It does not include the multimodal image or audio towers from the full Gemma 4 release.

# Supported models

- `google/gemma-4-E2B`
- `google/gemma-4-E4B`

# Exporting the model

The exporter can download and convert the Hugging Face checkpoint automatically, or you can point it at a pre-converted local checkpoint.

## Export Gemma 4 E2B

```bash
PYTHONPATH=.:.. python examples/models/llama/export_llama.py \
  --model gemma4_e2b \
  --params examples/models/gemma4/config/e2b_config.json \
  --dtype-override bf16 \
  --output-dir ./gemma4_e2b_out
```

## Export Gemma 4 E4B

```bash
PYTHONPATH=.:.. python examples/models/llama/export_llama.py \
  --model gemma4_e4b \
  --params examples/models/gemma4/config/e4b_config.json \
  --dtype-override bf16 \
  --output-dir ./gemma4_e4b_out
```

## Export with KV cache and custom SDPA

```bash
PYTHONPATH=.:.. python examples/models/llama/export_llama.py \
  --model gemma4_e4b \
  --params examples/models/gemma4/config/e4b_config.json \
  --dtype-override bf16 \
  --use_kv_cache \
  --use_sdpa_with_kv_cache \
  --disable_dynamic_shape \
  --output-dir ./gemma4_e4b_kv_out
```

# Notes

- The Gemma 4 exporter uses the native ExecuTorch text runtime and the local `convert_weights.py` checkpoint conversion flow.
- In local source-tree workflows, `PYTHONPATH=.:..` makes both `examples.*` and `executorch.*` imports work consistently.
