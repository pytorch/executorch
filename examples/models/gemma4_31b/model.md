# Gemma 4 31B-IT — Architecture & Design Notes

Developer reference for `model.py` and the `quant/` package. For
export/build/run instructions see [README.md](README.md).

The model mirrors the `Gemma4ForConditionalGeneration` text stack from
HuggingFace transformers / vLLM, with the ExecuTorch customizations needed
for `torch.export(strict=True)`.

## Architecture

```
Input tokens (B, T)
    |
    v
Embedding (vocab=262144, dim=5376) -> *= sqrt(hidden_size)  (normalizer)
    |
    v
+--- Decoder Layer x60 -----------------------------------------+
|                                                                |
|  residual = x                                                  |
|  RMSNorm -> Attention (sliding | full) -> RMSNorm -> +residual |
|  residual = x                                                  |
|  RMSNorm -> MLP (gate_proj, up_proj, down_proj, GELU-tanh)     |
|         -> RMSNorm -> +residual                                |
|  x *= layer_scalar  (per-layer buffer)                         |
|                                                                |
+----------------------------------------------------------------+
    |
    v
RMSNorm -> LM Head (tied with embed) -> tanh(logits/30) * 30
    |
    v
Gumbel-max sample(temperature) -> next token (B, 1)
```

Layer pattern (`5 sliding + 1 full`, repeated 10x — the last layer is full):

```
S S S S S F  S S S S S F  ...  S S S S S F   (S = sliding, F = full)
```

## Attention details

Two attention flavors, selected by `config.layer_types[layer_idx]`:

| Property            | Sliding (50 layers) | Full (10 layers, idx 5,11,...,59) |
|---------------------|--------------------|-----------------------------------|
| `head_dim`          | 256                | 512                               |
| `num_kv_heads`      | 16                 | 4                                 |
| `num_heads`         | 32                 | 32                                |
| RoPE θ              | 10 000             | 1 000 000                         |
| RoPE flavor         | full neox          | proportional, partial=0.25        |
| K = V               | no                 | yes (no `v_proj`)                 |
| Causal mask         | causal             | causal                            |
| Window restriction  | 1024 tokens        | none                              |
| Q-norm / K-norm     | RMSNorm w/ weight  | RMSNorm w/ weight                 |
| V-norm              | RMSNorm no weight  | RMSNorm no weight                 |
| `scaling`           | 1.0                | 1.0                               |

Notes:

- **Proportional partial RoPE**: the inv_freq vector for full-attention layers
  has the first `head_dim * partial_rotary_factor / 2 = 64` frequencies real
  (computed with denominator `head_dim`, not `rotary_dim` — that's the
  proportional part) and the remaining `head_dim/2 - 64 = 192` zero so cos=1
  and sin=0 (identity rotation) for the non-rotated dims.
- **K = V**: on full-attention layers `v_proj` is absent in the checkpoint
  and `V` is taken from the pre-norm `K` projection. After `k_norm` /
  RoPE on K and `v_norm` (weightless) on V the two diverge, so the cache
  still stores them separately.
- **Mask construction**: a single boolean `(1, 1, T_q, T_kv)` mask is built
  once per forward at the model level — one for sliding (causal AND
  pos_q - pos_k < 1024), one for full (just causal). Layers pick whichever
  matches their type and pass it to `F.scaled_dot_product_attention(...,
  enable_gqa=True)`.
- **Gemma `scaling=1.0`**: unlike Gemma 2/3, Gemma 4 does not scale Q by
  `query_pre_attn_scalar`; QK-norm handles attention magnitude.

## Model parameters (text stack)

| Parameter                       | Value      |
|---------------------------------|------------|
| `vocab_size`                    | 262 144    |
| `hidden_size`                   | 5 376      |
| `intermediate_size`             | 21 504     |
| `num_hidden_layers`             | 60         |
| `num_attention_heads`           | 32         |
| `num_key_value_heads` (sliding) | 16         |
| `head_dim` (sliding)            | 256        |
| `num_global_key_value_heads`    | 4          |
| `global_head_dim`               | 512        |
| `sliding_window`                | 1024       |
| `rms_norm_eps`                  | 1e-6       |
| `final_logit_softcapping`       | 30.0       |
| `tie_word_embeddings`           | true       |
| `max_position_embeddings`       | 262 144    |

Decoder norms per layer: `input_layernorm`, `post_attention_layernorm`,
`pre_feedforward_layernorm`, `post_feedforward_layernorm` — all
`RMSNorm` (multiplies by `weight` directly, not `(1 + weight)`).

## Methods exported (`export.py`)

| Method    | Input                                                      | Output (sampled) |
|-----------|------------------------------------------------------------|------------------|
| `decode`  | tokens `(1, 1)` + input_pos `(1,)` + temperature `(1,)`    | `(1, 1)` float   |
| `prefill` | tokens `(1, T)` + input_pos `(T,)` + temperature `(1,)`, T∈[2, min(max_seq_len-1, 2×sliding_window)] | `(1, 1)` float   |

Both methods share the same KV-cache buffers via
`MemoryPlanningPass(share_mutable_buffers=True)` and
`emit_mutable_buffer_names=True`. The exported program performs Gumbel-max
sampling on-device and returns a single token ID per call so the C++ runner
only has to feed tokens.

Prefill length is capped to the ring-buffer KV cache size
(`2 × sliding_window`) to avoid duplicate wrapped indices in
`index_copy_`. The C++ runner chunks longer prompts automatically using
the `get_max_prefill_chunk` constant method. Chunked prefill produces
identical logits to sequential one-token-at-a-time prefill.

## Quantization

Three modules in `quant/`:

- **Recipe** (`recipe.py`): `QuantConfig` (bits, group_size, symmetric,
  method) + `QuantRule` (regex pattern, config, optional layer filter) +
  `QuantRecipe` (ordered rules, first match wins). Declares what to
  quantize and how — says nothing about packing or backends.
- **Serialize** (`serialize.py`): `CanonicalQuantizedWeight` (int8 qdata +
  bf16 scale + optional zero). `save()` / `load()` persist to safetensors
  with a JSON header per weight. Packing-agnostic — any backend can read
  the file.
- **Packer** (`pack_cuda.py`): converts `CanonicalQuantizedWeight` to
  backend runtime format at load time via `pack_model()`. Dispatches per
  parent module type (`nn.Linear` → `Int4TilePackedTo4dTensor` for
  tinygemm). Extensible via a packers dict.

The quantize-once flow:

```
quantize_and_save.py                    export.py / inference.py
     |                                       |
  bf16 weights                          quantized checkpoint (safetensors)
     |                                       |
  quantize_weight()                     load()
     |                                       |
  CanonicalQuantizedWeight              CanonicalQuantizedWeight
     |                                       |
  save()                                pack_model()
     |                                       |
  model.safetensors                     Int4TilePackedTo4dTensor (runtime)
```

`embed_tokens` and `lm_head` start tied; they are untied before
quantization so `lm_head` (a 5376→262 144 matmul, very expensive at decode)
gets quantized. The embedding gets INT8 per-axis quantization (nearly
lossless for index lookup).

## Runtime buffer materialization

After weight loading (via `pack_model()` or `from_hf_checkpoint()`), the
model's KV caches, RoPE tables, and scalar constants are still on the meta
device. `materialize_runtime_buffers(model, dtype, device)` in `model.py`
replaces them with real tensors:

- KV caches → zeros in `dtype` (bf16 for inference, bf16 for export)
- RoPE tables → computed per-layer (sliding vs full, different θ and head_dim)
- `embed_normalizer`, `logit_softcap`, `cache_positions` → scalar constants

Called by `export.py` (device="cpu" for tracing) and `inference.py`
(device="cuda" for eager execution). Having one function avoids duplicating
the RoPE computation and constant setup across scripts.

## Customizations vs. vLLM / transformers reference

These exist solely to make the model exportable / efficient under ExecuTorch:

- **Boolean attention mask** built once per forward and shared across layers
  of the same type, instead of HF's per-layer `_create_causal_mask`.
- **Ring-buffer KV cache** for sliding layers (`RingKVCache`, sized to
  `2 × sliding_window`) saves memory for long sequences — positions wrap
  via modulo and the attention mask reconstructs which slots are valid.
  Full-attention layers use a flat `Gemma4KVCache` sized to `max_seq_len`.
  Both use `index_copy_(dim=2, ...)` for trace-friendly updates.
- **Per-layer RoPE tables** registered as `persistent=False` buffers (sliding
  uses full RoPE, full uses proportional partial RoPE — head_dim and θ
  differ, so the table is not shared).
- **On-device Gumbel-max sampling** so the exported program emits a token
  rather than a full logits tensor — keeps the runner GPU↔CPU traffic to a
  single float per step.
- **Final-logit softcap baked into the graph**, applied before sampling.
- **Meta-device construction + assign-load** keeps peak memory small enough
  to load the 31B-parameter checkpoint on one machine.

## Shared primitives

The numerically-sensitive math primitives are imported from
`examples.models.gemma4.text_decoder` and shared with the Gemma 4 E2B/E4B
example: `RMSNorm`, `RMSNormNoWeight`, `Gemma4MLP`, `Gemma4KVCache`,
`precompute_freqs_cis`, `apply_rotary_emb`. The 31B-specific pieces
(attention with K=V branch, decoder layer, top-level model with softcap +
sampling, checkpoint loader) live in `model.py`.
