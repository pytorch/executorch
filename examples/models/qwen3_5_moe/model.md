# Qwen 3.5 MoE — Architecture & Design Notes

Developer reference for `model.py` and `export.py`. For export/usage
instructions see [README.md](README.md).

## Architecture

```
Input tokens
    |
    v
Token Embedding (no learned position embedding — RoPE is inside attention)
    |
    v
+--- Decoder Layer x40 -----------------------------------------+
|                                                                |
|  GemmaRMSNorm -> Attention (hybrid) -> residual add            |
|    +- 75% of layers: GatedDeltaNet (linear, O(n))              |
|    +- 25% of layers: Full Attention (softmax, O(n^2))          |
|                                                                |
|  GemmaRMSNorm -> Sparse MoE -> residual add                   |
|    +- Router: top-8 expert selection + softmax weights          |
|    +- 256 routed experts: independent SwiGLU MLPs              |
|    +- Shared expert: always-on SwiGLU with sigmoid gate        |
|                                                                |
+----------------------------------------------------------------+
    |
    v
GemmaRMSNorm -> LM Head -> logits
```

Layer pattern (`full_attention_interval=4`):

```
L L L F L L L F L L L F ... L L L F    (L = GatedDeltaNet, F = Full Attention)
```

## Model Parameters

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 40 |
| `num_attention_heads` / `num_kv_heads` | 16 / 2 |
| `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 (64 of 256 dims rotated) |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 32 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 |
| `num_experts` / `num_experts_per_tok` | 256 / 8 |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 512 |
| `vocab_size` | 248320 |
| Total parameters | ~35B (~3B active per token) |

## Key Components

| Component | Description |
|-----------|-------------|
| **GemmaRMSNorm** | `x / sqrt(mean(x^2) + eps) * (1 + weight)` — unit-offset variant, weight init to zeros |
| **RMSNormGated** | `weight * RMSNorm(x) * silu(z)` — used in GatedDeltaNet output |
| **Full Attention** | GQA with output gate (sigmoid), QK-norm (GemmaRMSNorm), partial RoPE (25% of dims). `q_proj` produces Q + gate (2x heads). |
| **GatedDeltaNet** | Linear attention via recurrent state. Mamba-style gating: `g = -exp(A_log) * softplus(a + dt_bias)`. Causal conv1d, L2-normalized Q/K, delta rule recurrence. Uses FLA Triton kernel on CUDA. |
| **Sparse MoE** | Router selects top-8 of 256 experts per token. Shared expert with sigmoid gate always runs. |

## Memory-Efficient Loading

`from_hf_checkpoint()` uses the voxtral_realtime pattern to minimize peak
memory (~1x model size instead of ~3x):

1. **Meta device construction** — `with torch.device("meta"):` builds the
   model with zero-storage parameter tensors (shape/dtype metadata only).
2. **safetensors lazy access** — `safe_open` loads tensors on demand from
   each shard, remapping checkpoint keys inline.
3. **`assign=True` state dict loading** — replaces meta tensors by reference
   instead of copying into pre-allocated storage. No duplication.
4. **Buffers stay on meta** — KV caches, conv/recurrent state, causal masks,
   and RoPE tables remain on meta device. They are materialized in
   `export.py` before `torch.export` (which requires real tensors for
   in-place buffer ops).

## Expert Weight Structure

Expert weights are stored as grouped `nn.Linear` modules for quantization
compatibility. Each group of 16 experts shares a single `nn.Linear`:

- `gate_up_projs[g]`: `nn.Linear(2048, 16 * 512 * 2)` — fused gate+up
- `down_projs[g]`: `nn.Linear(512, 16 * 2048)` — down projection

16 experts per group keeps each `nn.Linear` under ~32K output features,
within tinygemm int4 packing limits. 256 experts / 16 = 16 groups, giving
32 matmul nodes per layer instead of 768 with per-expert linears.

Forward pass: compute all groups → cat → gather top-k → SwiGLU → compute
all groups → cat → gather correct expert per slot.

## Quantization

`export.py` is split into `load_and_quantize()` and `export_and_lower()`.

Quantization is done layer-by-layer on CUDA: each layer's parameters (not
meta buffers) are moved to CUDA, quantized (tinygemm int4 packing requires
CUDA), then moved back to CPU. Peak GPU memory is ~1 bf16 layer at a time.
The model stays on CPU — `torch.export` traces the graph without executing
ops.

With `--qlinear 4w --qembedding 8w`:

| Component | Quantization |
|-----------|-------------|
| 40 layers (attention + MoE linears) | 4w (int4 weight-only) |
| `lm_head` | 4w |
| `embed_tokens` | 8w (int8 weight-only) |
| Conv1d, norm weights, `A_log`, `dt_bias` | unquantized (bf16) |

Embedding and lm_head are untied before quantization since they require
different quantization formats (embedding uses index lookup, lm_head uses
matmul).

## Weight Mapping

| Checkpoint prefix | Model prefix |
|-------------------|-------------|
| `model.embed_tokens.weight` | `embed_tokens.weight` |
| `model.norm.weight` | `norm.weight` |
| `model.layers.{N}.input_layernorm.weight` | `layers.{N}.ln_1.weight` |
| `model.layers.{N}.post_attention_layernorm.weight` | `layers.{N}.ln_2.weight` |
| `model.layers.{N}.self_attn.{q,k,v,o}_proj.weight` | `layers.{N}.attn.{q,k,v,o}_proj.weight` |
| `model.layers.{N}.self_attn.{q,k}_norm.weight` | `layers.{N}.attn.{q,k}_norm.weight` |
| `model.layers.{N}.linear_attn.*` | `layers.{N}.attn.*` |
| `model.layers.{N}.mlp.experts.gate_up_proj` | `layers.{N}.mlp.cond_ffn.gate_up_projs.{G}.weight` (split into groups) |
| `model.layers.{N}.mlp.experts.down_proj` | `layers.{N}.mlp.cond_ffn.down_projs.{G}.weight` (split into groups) |
| `model.layers.{N}.mlp.gate.weight` | `layers.{N}.mlp.gate.weight` |
| `model.layers.{N}.mlp.shared_expert.*` | `layers.{N}.mlp.shared_expert.*` |
| `model.layers.{N}.mlp.shared_expert_gate.weight` | `layers.{N}.mlp.shared_expert_gate.weight` |

Visual and MTP keys are skipped. `lm_head.weight` is cloned from
`embed_tokens.weight` if not present in checkpoint (tied embeddings).

## References

- [HF Transformers Qwen3.5 MoE](https://github.com/huggingface/transformers) — `transformers/models/qwen3_5_moe/`
- [vLLM Qwen3.5](https://github.com/vllm-project/vllm) — `vllm/model_executor/models/qwen3_5.py`
- [nano_qwen35_moe](https://github.com/mergennachin/nano_qwen35_moe/) — minimal reference implementation
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — the linear attention mechanism
