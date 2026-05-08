# Qwen 3.5 MoE

Self-contained ExecuTorch implementation of
[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B),
a ~35B total / ~3B active parameter Mixture-of-Experts language model.
Weights are loaded directly from the HuggingFace safetensors checkpoint.
Supports CUDA and MLX backends. See [model.md](model.md) for architecture and
implementation details.

## Overview

The pipeline has two stages: **export** (Python, once) and **inference**
(C++ runner, repeated). Export converts the HuggingFace checkpoint into a
`model.pte` file with int4 quantization. At inference time, the C++ runner
loads the `.pte`, `.ptd`, and a HuggingFace tokenizer, then generates text.

## Prerequisites

- ExecuTorch installed from source (see [building from source](../../../docs/source/using-executorch-building-from-source.md))
- [safetensors](https://pypi.org/project/safetensors/) (`pip install safetensors`)
- NVIDIA GPU with CUDA toolkit (tested on A100 80GB)
- Python dependencies: `pip install -r requirements.txt` (installs FLA / Flash Linear Attention)
- Model weights downloaded from HuggingFace. The directory should contain
  `config.json`, `model.safetensors.index.json`, safetensors shards, and
  `tokenizer.json`.

## Export

Export produces a `model.pte` and `aoti_cuda_blob.ptd` containing the
compiled CUDA kernels and quantized weights. Int4 quantization is
recommended — the model is too large to fit in VRAM at bf16.

```bash
python export.py \
    --model-id Qwen/Qwen3.5-35B-A3B \
    --output-dir ./qwen35_moe_exports \
    --qlinear 4w \
    --qembedding 8w
```

Or with a local directory:

```bash
python export.py \
    --model-dir ~/models/Qwen3.5-35B-A3B \
    --output-dir ./qwen35_moe_exports \
    --qlinear 4w \
    --qembedding 8w
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | (none) | HuggingFace model ID (e.g. `Qwen/Qwen3.5-35B-A3B`). Downloads automatically. |
| `--model-dir` | (none) | Local HuggingFace model directory with `config.json` + safetensors |
| `--output-dir` | `./qwen35_moe_exports` | Output directory |
| `--max-seq-len` | `4096` | KV cache length |
| `--qlinear` | (none) | Linear layer quantization: `4w`, `8w`, `8da4w`, `8da8w` |
| `--qlinear-group-size` | `32` | Group size for linear quantization |
| `--qembedding` | (none) | Embedding quantization: `8w` |
| `--hqq` | off | Use HQQ scale-only optimization for expert quantization (slower, better accuracy) |
| `--prequantized` | (none) | Path to prequantized bundle directory (skips quantization) |
| `--turboquant` | off | Enable TurboQuant TQ4 KV cache compression (3.8x cache savings) |

### TurboQuant KV Cache Compression

The `--turboquant` flag enables [TurboQuant](https://arxiv.org/abs/2504.19874)
KV cache compression (3.8x savings) on the 10 full-attention layers.

```bash
python export.py --prequantized qwen35_moe_int4_hqq --turboquant
```

### Prequantized Export

Quantization is slow (~30 min with HQQ). To avoid re-quantizing on every
export, use `quantize_and_save.py` to create a self-contained bundle, then
export from it:

```bash
# Step 1: Quantize once (slow)
python quantize_and_save.py \
    --model-dir ~/models/Qwen3.5-35B-A3B \
    --qlinear 4w \
    --qembedding 8w \
    --qlinear-group-size 128 \
    --hqq \
    --output qwen35_moe_int4_hqq

# Step 2: Export from bundle (fast, no --model-dir needed)
python export.py \
    --prequantized qwen35_moe_int4_hqq
```

The bundle contains `model.safetensors`, `config.json`, and tokenizer files.
It can be uploaded to HuggingFace Hub for easy sharing.

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
    --tokenizer_path ~/models/Qwen3.5-35B-A3B/tokenizer.json \
    --prompt "The meaning of life is" \
    --max_new_tokens 128
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | (required) | Path to exported `.pte` model |
| `--data_path` | (none) | Path to `.ptd` delegate data file (required for CUDA) |
| `--tokenizer_path` | (required) | Path to HuggingFace `tokenizer.json` |
| `--prompt` | `"Hello"` | Input prompt text |
| `--temperature` | `0.8` | Sampling temperature (0 = greedy) |
| `--max_new_tokens` | `128` | Maximum tokens to generate |

## Troubleshooting

- **OOM during export**: The model requires significant GPU memory even
  with int4 quantization. Try reducing `--max-seq-len` or using a GPU
  with more VRAM.
- **OOM during loading**: The 35B parameter model requires ~70 GB RAM to
  load from safetensors before quantization. Ensure sufficient system
  memory.
- **Missing `aoti_cuda_blob.ptd`**: This file is produced during export
  alongside the `.pte`. Both files are required for inference.

## MLX Backend (Apple Silicon)

The MLX backend enables running Qwen 3.5 MoE on Apple Silicon GPUs.
It replaces the Triton-dependent modules (FusedMoEExperts, GatedDeltaNet)
with MLX custom ops (`mlx::gather_qmm`, `mlx::gated_delta_rule`, `mlx::rope`).

### Export (MLX)

```bash
python export.py \
    --model-id Qwen/Qwen3.5-35B-A3B \
    --backend mlx \
    --qlinear 4w \
    --qlinear-group-size 64 \
    --output-dir ./qwen35_moe_mlx
```

Or with a local directory:

```bash
python export.py \
    --model-dir ~/models/Qwen3.5-35B-A3B \
    --backend mlx \
    --qlinear 4w \
    --qlinear-group-size 64 \
    --output-dir ./qwen35_moe_mlx
```

### MLX Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend mlx` | `cuda` | Use MLX backend for Apple Silicon |
| `--model-id` | (none) | HuggingFace model ID (downloads automatically) |
| `--model-dir` | (none) | Local model directory |
| `--qlinear` | (none) | Linear layer quantization: `4w`, `8w` |
| `--qlinear-group-size` | `32` | Group size (64 recommended for MLX) |
| `--qembedding` | (none) | Embedding quantization: `8w` |
| `--tiny-test` | off | Build tiny model with random weights for CI testing |

### Run (MLX)

```bash
python -m executorch.examples.models.qwen3_5_moe.run \
    --pte ./qwen35_moe_mlx/model.pte \
    --tokenizer Qwen/Qwen3.5-35B-A3B \
    --prompt "What is the capital of France?" \
    --max-new-tokens 50
```

### Tiny Model Test

For CI or quick pipeline validation (no model download needed):

```bash
# Export tiny model (~1 MB, random weights)
python export.py \
    --tiny-test \
    --backend mlx \
    --qlinear 4w \
    --qlinear-group-size 32 \
    --output-dir /tmp/qwen35_moe_mlx_tiny

# Run inference (random tokens, no tokenizer needed)
python -m executorch.examples.models.qwen3_5_moe.run \
    --pte /tmp/qwen35_moe_mlx_tiny/model.pte \
    --prompt-len 4 \
    --max-new-tokens 5
```
