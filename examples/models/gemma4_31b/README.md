# Gemma 4 31B-IT

Text-only export of Google's Gemma 4 31B-IT to ExecuTorch with INT4/INT8
weight quantization. Currently supports the CUDA backend.

For architecture and design notes see [model.md](model.md).

## When to use which script

The full bf16 weights for 31B (~62 GB) often don't fit in available RAM. The
recommended flow is to quantize once and reuse the quantized checkpoint for
both export and eager inference:

| Script | Purpose | Peak memory |
|---|---|---|
| `quantize_and_save.py` | bf16 HF checkpoint → quantized checkpoint (one-time) | ~30 GB CPU |
| `export.py --prequantized <dir>` | quantized checkpoint → `model.pte` + `model.ptd` | ~24 GB CPU + CUDA for packing |
| `inference.py --prequantized <dir>` | quantized checkpoint → eager generation under `torch.compile` | ~24 GB GPU |
| `inference.py --gguf <file>` | GGUF file (Q4_K_M, etc.) → eager generation | ~24 GB GPU |
| `export.py --model-dir <hf>` | one-shot bf16 → quantize → export (no intermediate file) | ~30 GB CPU + CUDA for packing |

The quantized checkpoint is a safetensors file with int values + per-group
scales and a JSON header describing each weight's `QuantConfig`. No tensor
subclass or backend-specific packing — packing for the target backend happens
at load time via `quant.pack_model()`.

## Quantization recipes

Two built-in recipes (see `quantize_and_save.py`):

| Recipe | Description |
|---|---|
| `default` | INT4 min_max linears, INT8 per-axis embedding |
| `sensitive` | INT8 for edge-layer v_proj/down_proj, INT4 hqq elsewhere, INT8 per-axis embedding |

## Prequantized checkpoint

A prequantized checkpoint (sensitive recipe) is available on HuggingFace:

```bash
huggingface-cli download SocialLocalMobile/gemma-4-31B-it-HQQ-INT4 --local-dir gemma-4-31B-it-HQQ-INT4
```

> **Note**: This checkpoint is intended for development and testing of the
> ExecuTorch CUDA export pipeline. Output quality has not been formally
> evaluated against the base model.

Use it directly with `--prequantized` in the export and inference scripts
below — no need to run `quantize_and_save.py`.

## Quantize from scratch (optional)

To quantize from the original bf16 checkpoint instead, pass
`--quant-recipe` to select a recipe (`default` or `sensitive`):

```bash
python examples/models/gemma4_31b/quantize_and_save.py \
    --model-dir /path/to/gemma-4-31B-it \
    --output ./gemma4_31b_int4 \
    --quant-recipe sensitive
```

See [Quantization recipes](#quantization-recipes) above for details on each
recipe. Writes `model.safetensors`, `config.json`, and `tokenizer.json` into
`--output`.

## Export to ExecuTorch

```bash
python examples/models/gemma4_31b/export.py \
    --prequantized ./gemma4_31b_int4 \
    --output-dir ./gemma4_31b_exports \
    --max-seq-len 4096 \
    --backend cuda
```

Writes `model.pte` and `model.ptd` into `--output-dir`.

## Eager inference

```bash
python examples/models/gemma4_31b/inference.py \
    --prequantized ./gemma4_31b_int4 \
    --prompt "Write a short joke about saving RAM." \
    --max-new-tokens 128 \
    --temperature 0.8
```

GGUF files from the community (e.g., Q4_K_M) can also be used directly:

```bash
python examples/models/gemma4_31b/inference.py \
    --gguf ./gemma-4-31B-it-Q4_K_M.gguf \
    --tokenizer-path /path/to/tokenizer.json \
    --prompt "Hello"
```

Useful before spending the export+lowering time to confirm the quantized
model produces sensible text.

## Build the runner

```bash
make gemma4_31b-cuda
```

The binary lands at `cmake-out/examples/models/gemma4_31b/gemma4_31b_runner`.

## Run the .pte

```bash
./gemma4_31b_runner \
    --model_path  ./gemma4_31b_exports/model.pte \
    --data_path   ./gemma4_31b_exports/aoti_cuda_blob.ptd \
    --tokenizer_path ./gemma4_31b_int4/tokenizer.json \
    --prompt "Write a short joke about saving RAM." \
    --max_new_tokens 128 \
    --temperature 0.8
```

For benchmarking, add `--cuda_graph` to capture the decode method in a CUDA
graph (decode is fully static — `T=1`).
