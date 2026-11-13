## Summary
[LFM3](https://huggingface.co/LiquidAI/LFM3-1.7B) is Liquid AI's hybrid model family: short
convolution layers (width 7) interleaved with full and sliding-window (256) attention layers,
with QK norm, a per-head sigmoid output gate, partial RoPE on full attention, and sandwich norms.

QNN and the generic export share `convert_weights.py` and `config/1_7b_config.json`.
Both consume split `B_proj`, `C_proj`, `x_proj` weights and the same permuted Q/K
weights. The shared config enables QNN post norms and generic sandwich norms.

LFM3 reuses the Llama example code (`llama_transformer.py`) with the LFM2 short-conv blocks.
The weight converter maps LFM3's "zero-tail" partial RoPE onto the prefix layout used by
`hf_apply_rotary_emb` by permuting the q/k head dims of full-attention layers; the full-attention
`rope_theta` in the config is therefore `1e6 ** 0.5`.

## Export to XNNPACK (8da4w)
```
python -m extension.llm.export.export_llm \
  --config examples/models/lfm3/config/lfm3_xnnpack_q8da4w.yaml \
  +base.model_class="lfm3_1_7b" \
  +base.params="examples/models/lfm3/config/1_7b_config.json" \
  +export.max_seq_length=256 \
  +export.max_context_length=1024 \
  +export.output_name="lfm3_1_7b_8da4w.pte"
```
`max_seq_length` (the largest prefill chunk) must not exceed the sliding window (256): the
sliding layers keep a ring buffer of `2 * sliding_window` positions, and a longer chunk would
overwrite keys that its own first positions still attend to.

The full-attention layers have a fixed context limit: use one conversation per runner invocation and keep prompt tokens plus requested new tokens at or below `max_context_length` (1024 in this recipe). The sliding layers do not make the full-attention cache unlimited. The runner caps prefill chunks to the exported token-input bound without changing the metadata used for context-capacity checks.

Run with `llama_main --max_new_tokens 128 --num_bos 0`, passing an explicit BOS and chat template in the prompt. Keep prompt tokens plus the requested new tokens within the configured context limit:
```
<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>
```
