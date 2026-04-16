# Qwen 3.5 MoE — Architecture & Implementation Reference

Developer reference for the qwen3_5_moe example. For export/usage
instructions see [README.md](README.md).

## Architecture

```
Input tokens
    |
    v
embed_tokens: nn.Embedding(248320, 2048)
    |
    v
+--- Block x40 (layers[i]) ----------------------------------------+
|                                                                    |
|  ln_1: GemmaRMSNorm(2048) -> attn -> residual add                 |
|    +- 30 layers (i % 4 != 3): GatedDeltaNet (linear, O(n))        |
|    +- 10 layers (i % 4 == 3): FullAttention (softmax, O(n^2))     |
|                                                                    |
|  ln_2: GemmaRMSNorm(2048) -> mlp: SparseMoE -> residual add       |
|    +- gate: nn.Linear(2048, 256) -> top-8 + softmax                |
|    +- experts: FusedMoEExperts (256 routed SwiGLU experts)          |
|    +- shared_expert: SwiGLU(2048, 512) always-on                   |
|    +- shared_expert_gate: nn.Linear(2048, 1) -> sigmoid            |
|                                                                    |
+--------------------------------------------------------------------+
    |
    v
norm: GemmaRMSNorm(2048) -> lm_head: nn.Linear(2048, 248320) -> logits
```

Layer pattern (`full_attention_interval=4`):
```
L L L F L L L F L L L F ... L L L F    (L = GatedDeltaNet, F = FullAttention)
0 1 2 3 4 5 6 7 8 9 ...            39
```

## Model Config

| Field | Value | Notes |
|-------|-------|-------|
| `hidden_size` | 2048 | |
| `num_hidden_layers` | 40 | 30 linear + 10 full attention |
| `num_attention_heads` | 16 | Full attention Q heads |
| `num_kv_heads` | 2 | Full attention KV heads (GQA 8:1) |
| `head_dim` | 256 | |
| `partial_rotary_factor` | 0.25 | 64 of 256 dims get RoPE |
| `rope_theta` | 10,000,000 | |
| `linear_num_key_heads` | 16 | GatedDeltaNet K heads |
| `linear_num_value_heads` | 32 | GatedDeltaNet V heads (head_repeat=2) |
| `linear_key_head_dim` | 128 | |
| `linear_value_head_dim` | 128 | |
| `linear_conv_kernel_dim` | 4 | Causal depthwise conv1d kernel |
| `num_experts` | 256 | |
| `num_experts_per_tok` | 8 | Top-k routing |
| `moe_intermediate_size` | 512 | Per-expert hidden dim |
| `shared_expert_intermediate_size` | 512 | |
| `vocab_size` | 248320 | |
| `rms_norm_eps` | 1e-6 | |
| Total parameters | ~35B | ~3B active per token |

## Component Details

### GemmaRMSNorm

Unit-offset RMSNorm. Weight initialized to zeros, formula uses `(1 + weight)`:
```python
normed = x / sqrt(mean(x^2) + eps)
return normed * (1.0 + weight)
```

### RMSNormGated

Used only in GatedDeltaNet output. Combines RMSNorm with SiLU gating:
```python
return (weight * RMSNorm(x)) * silu(z)
```

### FullAttention

GQA with output gate, QK-norm, and partial RoPE.

**Submodules:**
- `qkv_proj`: `nn.Linear(2048, 9216)` — fused Q (with gate) + K + V
  - Q + gate: `n_heads * head_dim * 2 = 16 * 256 * 2 = 8192`
  - K: `n_kv_heads * head_dim = 2 * 256 = 512`
  - V: `n_kv_heads * head_dim = 2 * 256 = 512`
- `o_proj`: `nn.Linear(4096, 2048)` — output projection
- `q_norm`, `k_norm`: `GemmaRMSNorm(256)` — applied before RoPE
- `rotary_emb`: partial RoPE on first 64 of 256 dims
- `kv_cache`: `KVCache` — `[1, 2, max_seq_len, 256]` for K and V
- `cache_positions`: `arange(max_seq_len)` — for causal mask

**Forward (decode, T=1):**
```
x -> qkv_proj -> split Q+gate, K, V
Q, K -> q_norm, k_norm -> partial RoPE
K, V -> kv_cache.update
Q, K_cached, V_cached -> SDPA (split-K or tiled) -> output
output * sigmoid(gate) -> o_proj
```

**Forward (prefill, T>1):** Same but uses `sdpa` instead of `sdpa_decode_splitk`.

### GatedDeltaNet

Linear attention with delta rule recurrence. Mamba-style gating.

**Submodules:**
- `in_proj`: `nn.Linear(2048, 12352)` — fused projection, split into:
  - `qkv` (conv_dim = k_dim*2 + v_dim = 2048*2 + 4096 = 8192): goes through conv1d
  - `z` (value_dim = 4096): gating signal for output norm
  - `b` (num_v_heads = 32): beta for delta rule
  - `a` (num_v_heads = 32): decay parameter
- `conv1d`: depthwise `Conv1d(8192, 8192, kernel=4, groups=8192)` — no bias
- `dt_bias`: `Parameter([32])` — bias for decay computation
- `A_log`: `Parameter([32])` — log of decay base
- `norm`: `RMSNormGated(128)` — output norm with SiLU gate
- `out_proj`: `nn.Linear(4096, 2048)`
- `conv_state`: buffer `[1, 8192, 4]` — causal conv1d state
- `recurrent_state`: buffer `[1, 32, 128, 128]` — delta rule state (H, K, V)

**Forward (decode, T=1):**
```
x -> in_proj -> split qkv, z, b, a
qkv -> causal conv1d (manual, with state) -> silu -> split Q, K, V
Q, K -> L2 normalize -> repeat_interleave (16 heads -> 32 heads)
beta = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)
state = state * exp(g)                    # decay
Sk = einsum(state, k)                     # project state by key
delta = beta * (v - Sk)                   # delta rule
state = state + einsum(k, delta)          # update state
output = einsum(state, q) * scale         # query state
output -> RMSNormGated(output, z) -> out_proj
```

**Forward (prefill, T>1):** Uses chunked FLA Triton kernel
`torch.ops.triton.chunk_gated_delta_rule` instead of the recurrent loop.

**State reset:** When `input_pos[0] == 0`, both `conv_state` and
`recurrent_state` are zeroed (multiplied by 0).

### FusedMoEExperts

Stores all expert weights as stacked tensors.

**Before quantization (nn.Parameters):**
- `w1_weight`: `[256, 1024, 2048]` — fused gate+up (2 * 512 = 1024)
- `w2_weight`: `[256, 2048, 512]` — down projection

**After quantization (registered buffers):**
- `w1`: `[256, 1024, 1024]` int8 — packed INT4 (two values per byte)
- `w1_scale`: `[256, 1024, 2048//gs]` bf16
- `w2`: `[256, 2048, 256]` int8 — packed INT4
- `w2_scale`: `[256, 2048, 512//gs]` bf16
- `group_size`: int — inferred from weight/scale shape ratio

**INT4 packing:** `uint4 = int4 + 8` (shift to [0,15]), then
`packed = low_nibble | (high_nibble << 4)` stored as int8.

**Forward (decode):** `torch.ops.triton.fused_moe` — vec-mat MoE kernel.
**Forward (prefill):** `torch.ops.triton.fused_moe_batched_gemm` — batched
tensor-core MoE kernel. Toggled via `use_batched_moe` flag.

### SparseMoE

```python
scores = gate(x)                              # [B*T, 256]
expert_weights, expert_indices = topk(scores, 8)
expert_weights = softmax(expert_weights)       # normalize top-k
routed_out = experts(x, expert_weights, expert_indices, top_k=8)
shared_out = shared_expert(x)                  # SwiGLU always runs
shared_gate_val = sigmoid(shared_expert_gate(x))
output = routed_out + shared_gate_val * shared_out
```

### SwiGLU

Used for shared expert. Fused gate+up projection:
```python
gate_up = gate_up_proj(x)            # [B, 2*intermediate]
gate, up = split(gate_up)
return down_proj(silu(gate) * up)
```

## State Buffers

All stateful buffers are registered buffers with in-place updates (no
state in/out function args). Shared across decode/prefill methods via
`share_mutable_buffers=True` in ExecuTorch export.

| Buffer | Shape | Per | Purpose |
|--------|-------|-----|---------|
| `kv_cache.k_cache` | `[1, 2, max_seq_len, 256]` | full_attn layer (10) | Key cache |
| `kv_cache.v_cache` | `[1, 2, max_seq_len, 256]` | full_attn layer (10) | Value cache |
| `conv_state` | `[1, 8192, 4]` | GDN layer (30) | Causal conv1d state |
| `recurrent_state` | `[1, 32, 128, 128]` | GDN layer (30) | Delta rule recurrent state |
| `cache_positions` | `[max_seq_len]` | full_attn layer (10) | For causal mask computation |

## Memory-Efficient Loading

`from_hf_checkpoint()` minimizes peak memory (~1x model size):

1. **Meta device construction** — `with torch.device("meta"):` allocates
   no storage.
2. **safetensors lazy access** — `safe_open` loads one shard at a time,
   remapping checkpoint keys inline via `_process_checkpoint_key`.
3. **Weight fusion** — separate Q/K/V projections are concatenated into
   fused `qkv_proj`; GDN projections fused into `in_proj`; shared expert
   gate+up fused into `gate_up_proj`. Done in `_fuse_projection_weights`.
4. **Expert stacking** — per-expert weights `experts.{N}.{gate,up,down}_proj`
   are stacked into `[E, N, K]` tensors. Fused format
   `experts.gate_up_proj` / `experts.down_proj` loaded directly.
5. **`assign=True` state dict loading** — replaces meta tensors by
   reference, no duplication.
6. **Buffers stay on meta** — KV caches, conv/recurrent state, masks, and
   RoPE tables materialized later in `_materialize_buffers`.

## Weight Mapping (HuggingFace -> Model)

Checkpoint keys may have `model.language_model.` prefix (multimodal
config). This is stripped in `_process_checkpoint_key`.

### Embeddings and head
| Checkpoint | Model |
|------------|-------|
| `model.embed_tokens.weight` | `embed_tokens.weight` |
| `model.norm.weight` | `norm.weight` |
| `lm_head.weight` | `lm_head.weight` (cloned from `embed_tokens` if absent) |

### Per-layer norms
| Checkpoint | Model |
|------------|-------|
| `model.layers.{N}.input_layernorm.weight` | `layers.{N}.ln_1.weight` |
| `model.layers.{N}.post_attention_layernorm.weight` | `layers.{N}.ln_2.weight` |

### Full attention (layers where N % 4 == 3)
| Checkpoint | Model |
|------------|-------|
| `model.layers.{N}.self_attn.q_proj.weight` | fused into `layers.{N}.attn.qkv_proj.weight` |
| `model.layers.{N}.self_attn.k_proj.weight` | fused into `layers.{N}.attn.qkv_proj.weight` |
| `model.layers.{N}.self_attn.v_proj.weight` | fused into `layers.{N}.attn.qkv_proj.weight` |
| `model.layers.{N}.self_attn.o_proj.weight` | `layers.{N}.attn.o_proj.weight` |
| `model.layers.{N}.self_attn.q_norm.weight` | `layers.{N}.attn.q_norm.weight` |
| `model.layers.{N}.self_attn.k_norm.weight` | `layers.{N}.attn.k_norm.weight` |

### GatedDeltaNet (layers where N % 4 != 3)
| Checkpoint | Model |
|------------|-------|
| `model.layers.{N}.linear_attn.in_proj_qkv.weight` | fused into `layers.{N}.attn.in_proj.weight` |
| `model.layers.{N}.linear_attn.in_proj_z.weight` | fused into `layers.{N}.attn.in_proj.weight` |
| `model.layers.{N}.linear_attn.in_proj_b.weight` | fused into `layers.{N}.attn.in_proj.weight` |
| `model.layers.{N}.linear_attn.in_proj_a.weight` | fused into `layers.{N}.attn.in_proj.weight` |
| `model.layers.{N}.linear_attn.conv1d.weight` | `layers.{N}.attn.conv1d.weight` |
| `model.layers.{N}.linear_attn.dt_bias` | `layers.{N}.attn.dt_bias` |
| `model.layers.{N}.linear_attn.A_log` | `layers.{N}.attn.A_log` |
| `model.layers.{N}.linear_attn.norm.weight` | `layers.{N}.attn.norm.weight` |
| `model.layers.{N}.linear_attn.out_proj.weight` | `layers.{N}.attn.out_proj.weight` |

### MoE
| Checkpoint | Model |
|------------|-------|
| `model.layers.{N}.mlp.gate.weight` | `layers.{N}.mlp.gate.weight` |
| `model.layers.{N}.mlp.shared_expert_gate.weight` | `layers.{N}.mlp.shared_expert_gate.weight` |
| `model.layers.{N}.mlp.shared_expert.gate_proj.weight` | fused into `layers.{N}.mlp.shared_expert.gate_up_proj.weight` |
| `model.layers.{N}.mlp.shared_expert.up_proj.weight` | fused into `layers.{N}.mlp.shared_expert.gate_up_proj.weight` |
| `model.layers.{N}.mlp.shared_expert.down_proj.weight` | `layers.{N}.mlp.shared_expert.down_proj.weight` |
| `model.layers.{N}.mlp.experts.gate_up_proj` | `layers.{N}.mlp.experts.w1_weight` [E, 2*I, H] |
| `model.layers.{N}.mlp.experts.down_proj` | `layers.{N}.mlp.experts.w2_weight` [E, H, I] |
| `model.layers.{N}.mlp.experts.{E}.gate_proj.weight` | stacked into `w1_weight` (alt format) |
| `model.layers.{N}.mlp.experts.{E}.up_proj.weight` | stacked into `w1_weight` (alt format) |
| `model.layers.{N}.mlp.experts.{E}.down_proj.weight` | stacked into `w2_weight` (alt format) |

Ignored keys: `rotary_emb.inv_freq`, `linear_attn.conv1d.bias`,
visual/MTP prefixed keys.

## Quantization

### Standard mode (`_quantize`, `--qlinear 4w --qembedding 8w`)

Uniform INT4 — works for models with quantization-aware training (Qwen 3.5).

| Component | Method | Format |
|-----------|--------|--------|
| Expert w1/w2 | `_quantize_experts_int4` | Packed INT4 buffers + bf16 scales |
| All `nn.Linear` in layers | `quantize_model_("4w")` | `Int4TilePackedTo4dTensor` (tinygemm) |
| `lm_head` | `quantize_model_("4w")` | `Int4TilePackedTo4dTensor` |
| `embed_tokens` | `quantize_model_(qembedding="8w")` | `IntxUnpackedToInt8Tensor` |
| Norms, conv1d, dt_bias, A_log | Unquantized | bf16 |

Layer-by-layer on CUDA: move layer to GPU, quantize, move back to CPU.
Peak GPU memory ~1 layer at a time.

### Sensitive mode (`_quantize_sensitive`, `--sensitive`)

Mixed-precision — required for models without QAT (Qwen 3.6).
Determined by per-layer error profiling and GGUF Q4_K_M analysis.

| Component | Method | Format | bpw (gs=32) |
|-----------|--------|--------|-------------|
| Expert w1/w2 | `_quantize_experts_int4` | Packed INT4 + bf16 scales | 4.5 |
| Attention projections | `quantize_model_("8w")` | `IntxUnpackedToInt8Tensor` | 8.5 |
| Shared expert | `quantize_model_("8w")` | `IntxUnpackedToInt8Tensor` | 8.5 |
| `lm_head` | `quantize_model_("8w")` | `IntxUnpackedToInt8Tensor` | 8.5 |
| `embed_tokens` | `quantize_model_(qembedding="8w")` | `IntxUnpackedToInt8Tensor` | 8.5 |
| MoE gate, shared expert gate | Unquantized | bf16 | 16 |
| GDN conv1d, dt_bias, A_log, norm | Unquantized | bf16 | 16 |
| Layer norms, QK norms, final norm | Unquantized | bf16 | 16 |

Selective quantization uses `nn.ModuleDict` wrappers to pass only
specific submodules to `quantize_model_`, leaving the rest at bf16.

`--hqq` enables HQQ (Half-Quadratic Quantization) for expert INT4 —
iterative least-squares scale refinement. Only affects expert w1/w2,
not the INT8 layers.

### Prequantized checkpoints (`quantize_and_save.py`)

Saves quantized model to safetensors for fast reload via
`--prequantized`. Tensor subclasses (`Int4TilePackedTo4dTensor`,
`IntxUnpackedToInt8Tensor`) are flattened into inner tensors with
`.__qdata`, `.__scale`, `.__scale_and_zero`, `.__zero_point` suffixes.
Reconstruction metadata stored in safetensors header under `"quantization"`.

`load_prequantized_model` reconstructs subclasses via
`__tensor_unflatten__`, replaces `FusedMoEExperts` parameters with
quantized buffers, and infers `group_size` from weight/scale shape ratio.

## Export

`export_and_lower()` produces two methods sharing mutable state buffers:

| Method | Shape | MoE kernel | Use |
|--------|-------|------------|-----|
| `decode` | T=1, static | `fused_moe` (vec-mat) | Token-by-token generation |
| `prefill` | T>=2, dynamic | `fused_moe_batched_gemm` (tensor-core) | Prompt processing |

Both share KV cache, conv_state, and recurrent_state via
`share_mutable_buffers=True`. The prefill example uses
`T=max_seq_len-1` so AOTI compiles kernels for the full sequence range.

Output: `model.pte` (program) + `aoti_cuda_blob.ptd` (CUDA kernels + weights).

## Implementation Gotchas

Things that will break if you change them without understanding why:

### Manual conv1d implementation
GatedDeltaNet implements conv1d as a manual loop over kernel taps instead
of using `nn.Conv1d.forward()`. This is because `torch.export` decomposes
`nn.Conv1d` into `conv2d` ops, which lack AOTI fallback kernels. The
manual loop produces simple `mul` + `add` ops that AOTI handles natively.
The `conv1d.weight` is still an `nn.Conv1d` module (for correct weight
loading), but only `.weight` is accessed directly in forward.

### `assign=True` in load_state_dict
Both `from_hf_checkpoint` and `load_prequantized_model` use
`model.load_state_dict(state_dict, strict=False, assign=True)`.
`assign=True` replaces meta tensors by reference — without it, PyTorch
tries to copy data into meta storage, which fails. For quantized models,
removing `assign=True` silently converts tensor subclasses
(`IntxUnpackedToInt8Tensor`, `Int4TilePackedTo4dTensor`) to regular
Parameters, losing quantization.

### Two expert checkpoint formats
HuggingFace checkpoints come in two formats for expert weights:
1. **Fused**: `model.layers.{N}.mlp.experts.gate_up_proj` — single
   `[E, 2*I, H]` tensor. Loaded directly as `w1_weight`.
2. **Per-expert**: `model.layers.{N}.mlp.experts.{E}.gate_proj.weight` —
   individual `[I, H]` tensors per expert. Stacked in
   `_load_and_remap_checkpoint` into `[E, 2*I, H]`.

Both produce the same `w1_weight`/`w2_weight` tensors. The format depends
on how the checkpoint was saved upstream. `_process_checkpoint_key`
handles both via `_FUSED_EXPERT_RE` and `_EXPERT_RE` regex patterns.

### `_to_device_skip_meta` for quantization
During quantization, layers are moved to CUDA one at a time. But some
submodules have meta-device buffers (KV cache, conv_state,
recurrent_state) that can't be moved. `_to_device_skip_meta` walks the
module tree and only moves submodules that have no meta buffers. Without
this, `layer.to("cuda")` crashes on meta buffers.

### `torch.split` vs slicing in GatedDeltaNet
The forward uses explicit slicing (`proj[..., :cd]`) instead of
`torch.split` because `torch.split` produces `split_copy` ops in the
export graph, which lack AOTI fallback. Slicing produces `slice` ops
that AOTI handles.

### Sensitive quantization wrapping pattern
`_quantize_sensitive` uses `nn.ModuleDict` wrappers to selectively
quantize specific submodules:
```python
wrapper = nn.ModuleDict({"attn": nn.ModuleDict({
    "in_proj": layer.attn.in_proj,
    "out_proj": layer.attn.out_proj,
})})
quantize_model_(wrapper, qlinear_config="8w", ...)
layer.attn.in_proj = wrapper.attn.in_proj
```
This is necessary because `quantize_model_` quantizes every `nn.Linear`
it finds. The wrapper exposes only the linears we want quantized, leaving
GDN internals (conv1d, dt_bias, A_log, norm) and routing gates at bf16.

## References

- [HF Transformers Qwen3.5 MoE](https://github.com/huggingface/transformers) — `transformers/models/qwen3_5_moe/`
- [vLLM Qwen3.5](https://github.com/vllm-project/vllm) — `vllm/model_executor/models/qwen3_5.py`
- [nano_qwen35_moe](https://github.com/mergennachin/nano_qwen35_moe/) — minimal reference implementation
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — the linear attention mechanism
