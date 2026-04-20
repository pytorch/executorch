# Gemma4 ExecuTorch — CPU + XNNPACK Implementation Plan

## Goal

Enable [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) to run
on CPU with XNNPACK backend in ExecuTorch: fp32, bf16, and quantized (8da4w).
Benchmark against llama.cpp with
[unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF).

## Gemma4 E2B Architecture

Source: `transformers==5.5.4` `Gemma4TextConfig` from HuggingFace.

| Property | Value |
|----------|-------|
| Layers | 35 (28 sliding + 7 full attention, pattern: every 5th is full) |
| Hidden size | 1536 |
| Heads / KV heads | 8 / 1 (GQA 8:1) |
| Head dim | 256 (sliding), 512 (full attention `global_head_dim`) |
| Intermediate | 6144 |
| Vocab | 262144 |
| Activation | `gelu_pytorch_tanh` |
| Sliding window | 512 |
| Max position | 131072 |
| Tie embeddings | Yes |
| RoPE | Dual: sliding `theta=10000` default; full `theta=1e6` with `partial_rotary_factor=0.25` |
| Logit softcapping | `final_logit_softcapping=30.0` |
| KV sharing (YOCO) | `num_kv_shared_layers=20` (last 20 layers share KV from earlier donors) |
| MoE | `enable_moe_block=False` for E2B (dense model, NOT MoE despite the name) |
| Per-layer input (PLI) | `hidden_size_per_layer_input=256` — bottleneck gate per layer |
| Double-wide MLP | `use_double_wide_mlp=True` on KV-shared layers (intermediate × 2) |
| V norm | `Gemma4RMSNorm(head_dim, with_scale=False)` on value projections |
| Multimodal | Vision encoder (768d, 16 layers, patch=16) + Audio encoder (1024d, 12 layers) |

## What ExecuTorch Already Supports

The `examples/models/llama/` infrastructure (used by Gemma3) handles most features:

| Feature | Status | Location |
|---------|--------|----------|
| GQA (8:1) | Supported | `model_args.py:n_kv_heads` |
| Sliding + full attention | Supported | `model_args.py:layer_types`, ring KV cache |
| QK norm | Supported | `model_args.py:use_qk_norm` |
| GELU activation | Supported | `model_args.py:act_fn` |
| Embedding scaling | Supported | `model_args.py:embedding_scale_factor` |
| Partial rotary | Supported | `model_args.py:partial_rotary_factor` |
| Final logit softcapping | Supported | `model_args.py:final_logit_softcapping` |
| KV sharing (YOCO) | Supported | `attention.py:num_kv_shared_layers` |
| Dual RoPE theta | Supported | `model_args.py:rope_theta` + `local_rope_theta` |
| XNNPACK partitioning | Supported | Export pipeline |

## What Needs New Implementation

### 1. Per-Layer Input (PLI) bottleneck — NEW ARCHITECTURE FEATURE

Gemma4 introduces a per-layer gating mechanism not present in Gemma3 or Llama:

```python
# After attention + MLP residual in each layer:
if hidden_size_per_layer_input:
    residual = hidden_states
    hidden_states = per_layer_input_gate(hidden_states)       # Linear(1536 → 256)
    hidden_states = gelu(hidden_states)
    hidden_states = hidden_states * per_layer_input            # element-wise with shared input
    hidden_states = per_layer_projection(hidden_states)        # Linear(256 → 1536)
    hidden_states = post_per_layer_input_norm(hidden_states)   # RMSNorm
    hidden_states = residual + hidden_states
```

The `per_layer_input` is the token embedding (after scaling) passed through all
layers as a skip connection. This requires:
- New `ModelArgs` field: `hidden_size_per_layer_input: int = 0`
- New weights per layer: `per_layer_input_gate`, `per_layer_projection`,
  `post_per_layer_input_norm`
- Modified `TransformerBlock.forward()` to accept and use `per_layer_input`
- Modified `Transformer.forward()` to pass embedding as `per_layer_input`

Implementation in: `llama_transformer.py` (or a Gemma4-specific override).

### 2. Different head dimensions per layer type

Gemma4 uses `head_dim=256` for sliding attention but `global_head_dim=512` for
full attention layers. This means Q/K/V projection sizes differ by layer type.
Currently `ModelArgs.head_dim` is global.

Options:
- A: Add `global_head_dim` to `ModelArgs`, let `AttentionMHA.__init__` pick
  based on `layer_type` (minimal change)
- B: Per-layer head_dim list in config (more general but more invasive)

Recommend option A.

### 3. Double-wide MLP on KV-shared layers

When `use_double_wide_mlp=True` AND the layer is a KV-shared layer, the MLP
intermediate size doubles (6144 → 12288). Requires:
- New `ModelArgs` field: `use_double_wide_mlp: bool = False`
- Modified `FeedForward.__init__` to check layer index vs
  `num_kv_shared_layers` and double intermediate size accordingly

### 4. Value normalization (`v_norm`)

Gemma4 applies `RMSNorm(head_dim, with_scale=False)` to value projections.
Not currently in ExecuTorch attention. Requires a small addition to
`AttentionMHA`.

### 5. Layer scalar

Each decoder layer has a `layer_scalar` buffer (initialized to 1.0) that
multiplies the layer output: `hidden_states *= self.layer_scalar`. Simple to
add.

### 6. Weight conversion (`convert_weights.py`)

New weight mappings needed beyond Gemma3:

```
model.language_model.layers.{i}.per_layer_input_gate.weight
model.language_model.layers.{i}.per_layer_projection.weight
model.language_model.layers.{i}.post_per_layer_input_norm.weight
model.language_model.layers.{i}.pre_feedforward_layernorm.weight
model.language_model.layers.{i}.post_feedforward_layernorm.weight
model.language_model.embed_tokens.weight  (with embed_scale)
```

Plus different Q/K/V projection sizes for sliding vs full attention layers.

## Implementation Plan

### Phase 1: Text-only model (no vision/audio)

Focus on the text decoder only — sufficient for the E2B benchmark.

#### 1.1 Create `examples/models/gemma4/`

```
examples/models/gemma4/
├── __init__.py              # Gemma4Model class
├── convert_weights.py       # HF → ExecuTorch weight mapping
├── config/
│   └── e2b_config.json      # Model config
├── CMakeLists.txt           # Build config (copy from gemma3, simplify)
├── CMakePresets.json         # gemma4-cpu preset
└── README.md                # Export/build/run instructions
```

#### 1.2 Extend `ModelArgs` and `llama_transformer.py`

Add to `model_args.py`:
```python
hidden_size_per_layer_input: int = 0
global_head_dim: Optional[int] = None
use_double_wide_mlp: bool = False
use_v_norm: bool = False
```

Modify `llama_transformer.py`:
- `TransformerBlock.forward()`: accept `per_layer_input` tensor, apply PLI
  gate after attention+MLP if `hidden_size_per_layer_input > 0`
- `Transformer.forward()`: pass scaled embedding as `per_layer_input`

Modify `attention.py`:
- `AttentionMHA.__init__`: use `global_head_dim` for full attention layers
- Add optional v_norm

Modify `feed_forward.py`:
- Support double-wide intermediate on KV-shared layers

#### 1.3 Write `convert_weights.py`

Map HF Gemma4 checkpoint names to ExecuTorch names. Key differences from
Gemma3:
- PLI weights (`per_layer_input_gate`, `per_layer_projection`, etc.)
- Different Q/K/V sizes for full vs sliding layers
- V norm weights
- Pre/post feedforward norm (4 norms per layer instead of 2)

#### 1.4 Write `e2b_config.json`

Translate the HF config to ExecuTorch `ModelArgs` JSON format.

#### 1.5 Export

Register Gemma4 in `export_llama_lib.py`:
```python
EXECUTORCH_DEFINED_MODELS["gemma4"] = "gemma4"
HUGGING_FACE_REPO_IDS["gemma4"] = "google/gemma-4-E2B"
```

Export commands:
```bash
# FP32 XNNPACK
python -m executorch.examples.models.llama.export_llama_lib \
  --model gemma4 --params examples/models/gemma4/config/e2b_config.json \
  --checkpoint <path>/model.safetensors \
  --use_sdpa_with_kv_cache -X -d fp32 \
  --local_global_attention --output gemma4.pte

# 8da4w quantized
python -m executorch.examples.models.llama.export_llama_lib \
  --model gemma4 --params examples/models/gemma4/config/e2b_config.json \
  --checkpoint <path>/model.safetensors \
  --use_sdpa_with_kv_cache -X -d fp32 \
  --pt2e_quantize xnnpack_dynamic_qc8 \
  --local_global_attention --output gemma4_q8.pte
```

#### 1.6 Build

Add to `Makefile`:
```makefile
gemma4-cpu:
    cmake --workflow --preset llm-release
    cd examples/models/gemma4 && cmake --workflow --preset gemma4-cpu
```

The C++ runner can use the standard LLM runner
(`extension/llm/runner/llm_runner.h`) — no custom runner needed for text-only.

#### 1.7 Run + benchmark

```bash
# ExecuTorch
./cmake-out/examples/models/gemma4/gemma4_runner \
  --model_path gemma4.pte \
  --tokenizer_path tokenizer.json \
  --prompt "What is the capital of France?"

# llama.cpp baseline
cd /home/younghan/llama.cpp
./llama-cli -m gemma-4-E2B-it-Q4_K_M.gguf \
  -p "What is the capital of France?" -n 100
```

Compare: tokens/sec, memory usage, output quality.

### Phase 2: Multimodal (future, not in scope)

Vision and audio encoders require separate export as additional .pte modules,
following the Gemma3 multimodal runner pattern. Defer to after text-only works.

## Risk assessment

| Risk | Mitigation |
|------|------------|
| PLI changes break existing models | Gate behind `hidden_size_per_layer_input > 0`; defaults to off |
| Different head_dim per layer type breaks export | Use separate Q/K/V projections per layer; torch.export handles dynamic shapes per module |
| XNNPACK doesn't support some op | Fall back to portable for that op; XNNPACK partitioner handles this automatically |
| KV sharing + ring buffer interaction | Already tested via YOCO in existing infra; Gemma4's sharing pattern (last 20 layers) maps directly |
| Weight download blocked by proxy | User downloads on local machine, rsync to devserver |

## Files to modify

| File | Change |
|------|--------|
| `examples/models/llama/model_args.py` | Add `hidden_size_per_layer_input`, `global_head_dim`, `use_double_wide_mlp`, `use_v_norm` |
| `examples/models/llama/llama_transformer.py` | PLI gate in `TransformerBlock`, pass embedding as `per_layer_input` |
| `examples/models/llama/attention.py` | `global_head_dim` support, v_norm |
| `examples/models/llama/feed_forward.py` | Double-wide MLP for KV-shared layers |
| `examples/models/llama/export_llama_lib.py` | Register `gemma4` model |
| `examples/models/gemma4/*` | New: model class, convert_weights, config, build, README |
| `Makefile` | Add `gemma4-cpu` target |
