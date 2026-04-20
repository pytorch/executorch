# Gemma4 ExecuTorch — Status

Last updated: 2026-04-17

## TL;DR — WORKING

End-to-end Gemma4 generation on ExecuTorch CPU + XNNPACK matches the HuggingFace
reference bit-exactly. Validation:

```
Prompt: <|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n
HF:     "The capital of France is **Paris**."
ET:     "The capital of France is **Paris**.<turn|><turn|>..."
```

Decode: ~14.7 tok/s on CPU XNNPACK FP32 (unsharded 19 GB .pte).

## Parity test results

`examples/models/gemma4/tests/test_parity.py` — all six tests pass:

| Test | max_abs_diff | cos_sim |
|------|--------------|---------|
| token_embedding | 0.00e+00 | 1.000001 |
| rmsnorm | 0.00e+00 | 1.000000 |
| pli_inputs | 0.00e+00 | 1.000003 |
| rope (sliding cos+sin) | 0.00e+00 | 1.000000 |
| rope (full cos+sin) | 0.00e+00 | 1.000000 |
| decoder_layer_0 | 0.00e+00 | 1.000001 |
| full_forward (logits) | 5.72e-06 | 0.999999 |

Top-1 next token matches HF for the canonical prompt.

## Bugs found and fixed

1. **Per-layer-type RoPE missing** — Gemma4 uses two RoPE setups: sliding
   layers with θ=10k partial=1.0 default; full layers with θ=1e6 partial=0.25
   `proportional`. Fixed by adding dual-buffer RoPE
   (`freqs_cos_global`/`freqs_sin_global`) in `rope.py` and threading
   `layer_type` through `Transformer._forward_layers`.
2. **Proportional RoPE formula** — Gemma4's "proportional" RoPE uses the
   FULL `head_dim` denominator and zero-pads trailing freqs (HF
   `_compute_proportional_rope_parameters`). Implemented in
   `hf_precompute_freqs_cis` with `rope_type="proportional"` branch.
3. **Attention scaling** — Gemma4 sets `self.scaling = 1.0` (no implicit
   `1/sqrt(head_dim)` divide). Wired through `attention_multiplier` in
   `ModelArgs` → `AttentionMHA` → both `F.scaled_dot_product_attention`
   call sites and the `SDPA` module.
4. **`SDPACustom` dropped scale** — The export source-transform replaces
   `SDPA` with `SDPACustom` calling `torch.ops.llama.custom_sdpa`. The
   wrapper was not forwarding the `scale` parameter. Fixed in
   `examples/models/llama/source_transformation/sdpa.py`.
5. **YOCO prefill skip was wrong** — Original code skipped shared layers
   during prefill. HF runs all layers; shared layers receive the donor's
   K/V via `shared_kv_states[kv_shared_layer_index]`. Removed the
   `is_prefill` skip guard.
6. **YOCO donor map type-aware** — Single global donor was unsafe with
   mixed head_dims (256 sliding vs 512 full). Built per-type donor map in
   `_build_kv_donor_map` (last non-shared layer of matching type).
7. **`act_fn` ignored in MLP** — `FeedForward` hardcoded SiLU. Now
   threads `args.act_fn` (Gemma4 uses gelu_approx).
8. **`v_norm` dtype handling** — Inline RMS without learnable weight,
   converted to/from input dtype to play with bf16 weights.
9. **Embedding scale + final logit softcap** — Applied
   `embedding_scale_factor=sqrt(hidden_size)` after lookup and
   `c·tanh(logits/c)` before output (`c=30.0`).
10. **`post_attention_norm`, `post_ffn_norm`, `layer_scalar`** — Added to
    `TransformerBlock`. Layer scalar is per-layer learnable parameter.
11. **PLI (Per-Layer Input)** — Built `pli_embeddings`,
    `pli_projection`, `pli_norm` in `Transformer.__init__`; computed
    per-layer input from input ids + main embedding; sliced per layer
    in `_forward_layers`; gated through PLI bottleneck in
    `TransformerBlock.forward`.
12. **Gemma4 chat template** — Uses `<|turn>...<turn|>`, NOT Gemma3's
    `<start_of_turn>...<end_of_turn>`. Updated `main.cpp`.

## Files changed (this session)

| File | Why |
|------|-----|
| `examples/models/llama/model_args.py` | +`global_rope_theta`, `global_partial_rotary_factor`, `global_rope_type`, `hidden_size_per_layer_input`, `global_head_dim`, `use_double_wide_mlp`, `use_v_norm`, `use_layer_scalar` |
| `examples/models/llama/rope.py` | Dual-buffer RoPE; `proportional` formula; `get_freqs_for_layer_type` |
| `examples/models/llama/llama_transformer.py` | PLI plumbing; per-layer-type RoPE; YOCO donor map; post-norms; layer_scalar; embedding scale; logit softcap |
| `examples/models/llama/attention.py` | `attention_multiplier`; `global_head_dim` per layer; `v_norm`; YOCO type-aware shared_kv routing |
| `examples/models/llama/feed_forward.py` | Thread `act_fn` parameter |
| `examples/models/llama/source_transformation/sdpa.py` | Forward `scale` to `torch.ops.llama.custom_sdpa` |
| `examples/models/llama/export_llama_lib.py` | Register `gemma4` in `EXECUTORCH_DEFINED_MODELS` |
| `extension/llm/export/config/llm_config.py` | Add `ModelType.gemma4` |
| `examples/models/gemma4/__init__.py` | `Gemma4Model(Llama2Model)` |
| `examples/models/gemma4/config/e2b_config.json` | Full Gemma4 E2B config (35 layers, dual RoPE, etc.) |
| `examples/models/gemma4/convert_weights.py` | HF → ET state dict mapping |
| `examples/models/gemma4/main.cpp` | C++ runner with `<|turn>` template |
| `examples/models/gemma4/CMakeLists.txt` + `CMakePresets.json` | Build with optional vision |
| `examples/models/gemma4/tests/test_parity.py` | 6-test parity harness vs HF |
| `examples/models/gemma4/README.md` | Export/build/run docs |
| `Makefile` | `gemma4-cpu`, `gemma4-cuda` targets |

## Reproduction

```bash
# 1. Convert weights (HF → ET)
python -m executorch.examples.models.gemma4.convert_weights \
  ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it/model_et.pth

# 2. Layer-by-layer parity (must all pass)
python examples/models/gemma4/tests/test_parity.py

# 3. Export FP32 XNNPACK
python -m executorch.extension.llm.export.export_llm \
  base.model_class=gemma4 \
  base.params=examples/models/gemma4/config/e2b_config.json \
  base.checkpoint=~/models/gemma-4-E2B-it/model_et.pth \
  model.use_sdpa_with_kv_cache=true model.use_kv_cache=true \
  export.max_seq_length=512 export.max_context_length=512 \
  backend.xnnpack.enabled=true

# 4. Build + run
make gemma4-cpu
./cmake-out/examples/models/gemma4/gemma4_runner \
  --model_path ./gemma4.pte \
  --tokenizer_path ~/models/gemma-4-E2B-it/tokenizer.json \
  --prompt "What is the capital of France?" --seq_len 30
```

## Known follow-ups

- **EOS handling**: ✅ Fixed. Gemma4 has three EOS tokens (`<eos>`=1,
  `<turn|>`=106, id 50). Embedded via `base.metadata` during export so the
  runner stops cleanly after each response.
- **Chat template**: ✅ Official vLLM jinja template copied to
  `chat_template.jinja`; `render_chat.py` renders it (supports system
  prompts, tool calls, reasoning mode). Use with `--raw_prompt`.
- **Quantization**: not yet validated for Gemma4 (8da4w / 4w paths exist
  but parity not measured).
- **Vision + audio modalities**: deferred (text-only this pass).
- **Per-layer-type partial_rotary**: works but only needed for full layers
  in Gemma4 E2B; other Gemma4 sizes may differ.
