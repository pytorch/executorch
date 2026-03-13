# Qwen 3.5 MoE

Self-contained ExecuTorch implementation of
[Qwen3.5-MoE-A3B](https://huggingface.co/Qwen/Qwen3.5-MoE-A3B),
a ~35B total / ~3B active parameter Mixture-of-Experts language model.
Weights are loaded directly from the HuggingFace safetensors checkpoint.
CUDA backend only.

## Overview

The pipeline has two stages: **export** (Python, once) and **inference**
(C++ runner, repeated). Export converts the HuggingFace checkpoint into a
`model.pte` file with int4 quantization. At inference time, the C++ runner
loads the `.pte`, `.ptd`, and a HuggingFace tokenizer, then generates text.

## Architecture

Qwen 3.5 MoE is a hybrid-attention transformer with sparse Mixture of Experts:

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
|    +- Router: softmax -> top-8 expert selection                |
|    +- 256 routed experts: independent SwiGLU MLPs              |
|    +- Shared expert: always-on SwiGLU with sigmoid gate        |
|                                                                |
+----------------------------------------------------------------+
    |
    v
GemmaRMSNorm -> LM Head -> logits
```

### Key parameters

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
| `vocab_size` | 248320 |

### Key components

| Component | Description |
|-----------|-------------|
| **GemmaRMSNorm** | `x / sqrt(mean(x^2) + eps) * (1 + weight)` — unit-offset variant |
| **Full Attention** | GQA with output gate (sigmoid), QK-norm (GemmaRMSNorm), partial RoPE (25% of dims) |
| **GatedDeltaNet** | Linear attention via recurrent state. Mamba-style gating: `g = -exp(A_log) * softplus(a + dt_bias)`. Causal conv1d, L2-normalized Q/K, delta rule recurrence. Uses FLA Triton kernel. |
| **Sparse MoE** | Router selects top-8 of 256 experts per token. Shared expert with sigmoid gate always runs. |

## Prerequisites

- ExecuTorch installed from source (see [building from source](../../../docs/source/using-executorch-building-from-source.md))
- [safetensors](https://pypi.org/project/safetensors/) (`pip install safetensors`)
- NVIDIA GPU with CUDA toolkit
- Model weights downloaded from HuggingFace. The directory should contain
  `config.json`, `model.safetensors.index.json`, safetensors shards, and
  `tokenizer.json`.

## Export

Export produces a `model.pte` and `aoti_cuda_blob.ptd` containing the
compiled CUDA kernels and quantized weights. Int4 quantization is
recommended — the model is too large to fit in VRAM at bf16.

```bash
python export.py \
    --model-dir ~/models/Qwen3.5-MoE-A3B \
    --output-dir ./qwen35_moe_exports \
    --qlinear 4w \
    --qlinear-packing-format tile_packed_to_4d
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | (required) | HuggingFace model directory with `config.json` + safetensors |
| `--output-dir` | `./qwen35_moe_exports` | Output directory |
| `--max-seq-len` | `4096` | KV cache length |
| `--qlinear` | (none) | Linear layer quantization: `4w`, `8w`, `8da4w`, `8da8w` |
| `--qlinear-group-size` | `32` | Group size for linear quantization |
| `--qlinear-packing-format` | (none) | Packing format for 4w: `tile_packed_to_4d` |
| `--qembedding` | (none) | Embedding quantization: `8w` |

## Build

ExecuTorch must be installed from source first (see
[Prerequisites](#prerequisites)). The `make` target handles building
core libraries and the runner binary.

```bash
make qwen3_5_moe-cuda
```

This builds ExecuTorch with CUDA backend support, then the runner binary
at `cmake-out/examples/models/qwen3_5_moe/qwen3_5_moe_runner`.

## Run

The runner requires:
- `model.pte` — exported model (see [Export](#export))
- `aoti_cuda_blob.ptd` — CUDA delegate data file (produced during export)
- `tokenizer.json` — HuggingFace tokenizer from the model weights directory

```bash
cmake-out/examples/models/qwen3_5_moe/qwen3_5_moe_runner \
    --model_path qwen35_moe_exports/model.pte \
    --data_path qwen35_moe_exports/aoti_cuda_blob.ptd \
    --tokenizer_path ~/models/Qwen3.5-MoE-A3B/tokenizer.json \
    --prompt "The meaning of life is" \
    --max_new_tokens 128
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | (required) | Path to exported `.pte` model |
| `--data_path` | (none) | Path to `.ptd` delegate data file (required for CUDA) |
| `--tokenizer_path` | (required) | Path to HuggingFace `tokenizer.json` |
| `--prompt` | `"Hello"` | Input prompt text |
| `--temperature` | `0.8` | Sampling temperature (0 = greedy) |
| `--max_new_tokens` | `128` | Maximum tokens to generate |

## Files

| File | Description |
|------|-------------|
| `model.py` | Export-friendly model definition with all components |
| `export.py` | Export + quantize + lower to CUDA `.pte` |
| `main.cpp` | C++ runner using ExecuTorch's TextLLMRunner |
| `CMakeLists.txt` | Build configuration |
| `CMakePresets.json` | CMake presets for CUDA build |

## Troubleshooting

- **OOM during export**: The model requires significant GPU memory even
  with int4 quantization. Try reducing `--max-seq-len` or using a GPU
  with more VRAM.
- **OOM during loading**: The 35B parameter model requires ~70 GB RAM to
  load from safetensors before quantization. Ensure sufficient system
  memory.
- **Missing `aoti_cuda_blob.ptd`**: This file is produced during export
  alongside the `.pte`. Both files are required for inference.
