## Summary
[Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) support in ExecuTorch is exported through the Llama example pipeline with a hybrid layer layout:
- `full_attention` layers use gated full attention.
- `linear_attention` layers use Gated DeltaNet with internal recurrent state.

This first bring-up is **fp32 + static shape** (`enable_dynamic_shape=False`).
Currently supported text model sizes: `0.8B`, `2B`, `4B`.

## Export
```bash
python -m extension.llm.export.export_llm \
  --config examples/models/qwen3_5/config/qwen3_5_xnnpack_fp32.yaml \
  +base.model_class="qwen3_5_0_8b" \
  +base.params="examples/models/qwen3_5/config/0_8b_config.json" \
  +export.output_name="qwen3_5_0_8b_fp32.pte"
```

```bash
python -m extension.llm.export.export_llm \
  --config examples/models/qwen3_5/config/qwen3_5_xnnpack_fp32.yaml \
  +base.model_class="qwen3_5_2b" \
  +base.params="examples/models/qwen3_5/config/2b_config.json" \
  +export.output_name="qwen3_5_2b_fp32.pte"
```

```bash
python -m extension.llm.export.export_llm \
  --config examples/models/qwen3_5/config/qwen3_5_xnnpack_fp32.yaml \
  +base.model_class="qwen3_5_4b" \
  +base.params="examples/models/qwen3_5/config/4b_config.json" \
  +export.output_name="qwen3_5_4b_fp32.pte"
```

The exporter will download and convert HF weights automatically when `+base.checkpoint` is not provided.
Install `safetensors` in your environment if it is missing:
```bash
python -m pip install safetensors
```

## Run (Python Runner)
```bash
python -m executorch.examples.models.llama.runner.native \
  --model qwen3_5_0_8b \
  --pte qwen3_5_0_8b_fp32.pte \
  --tokenizer /path/to/tokenizer.json \
  --tokenizer_config /path/to/tokenizer_config.json \
  --prompt "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n" \
  --params examples/models/qwen3_5/config/0_8b_config.json \
  --max_len 128 \
  -kv \
  --temperature 0.3
```

## Notes
- Current path targets CPU/XNNPACK export validation and runner compatibility.
- `q8da4w` quantization for Qwen3.5 is intentionally deferred to a follow-up.
- Dynamic-shape export is not enabled yet for Qwen3.5 DeltaNet layers in this path; keep `enable_dynamic_shape=False`.
- For static-shape exports, `runner.native` falls back to sequential token prefill for multi-token prompts.
- Default metadata uses Qwen3.5 special token ids: `get_bos_id=248045`, `get_eos_ids=[248046,248044]`.
