# Gemma 4 31B-IT

Export of Google's Gemma 4 31B-IT to ExecuTorch with INT4/INT8 weight
quantization. Supports CUDA (with multimodal text + vision) and MLX
(Apple Silicon, text-only) backends.

For architecture and design notes see [model.md](model.md).

## When to use which script

The full bf16 weights for 31B (~62 GB) often don't fit in available RAM. The
recommended flow is to quantize once and reuse the quantized checkpoint for
both export and eager inference:

| Script | Purpose | Image input? | Peak memory |
|---|---|---|---|
| `quantize_and_save.py` | bf16 HF checkpoint → quantized checkpoint (one-time) | — | ~30 GB CPU |
| `export.py --prequantized <dir>` | quantized checkpoint → `model.pte` + `model.ptd` (multimodal 4-method contract) | yes (runner-time) | ~24 GB CPU + CUDA for packing |
| `inference.py --prequantized <dir>` | quantized checkpoint → eager generation under `torch.compile` | `--image-path` | ~24 GB GPU |
| `inference.py --gguf <file>` | GGUF text decoder + auto-loaded vision tower → eager generation | `--image-path` | ~24 GB GPU |
| `inference.py --bf16 <dir>` | full bf16 HF safetensors → eager generation (debug only; ~62 GB) | `--image-path` | ~62 GB CPU + ~62 GB GPU |
| `export.py --model-dir <hf>` | one-shot bf16 → quantize → export (no intermediate file) | — (text-only path) | ~30 GB CPU + CUDA for packing |

The quantized checkpoint is a safetensors file containing torchao tensor
subclasses (`Int4Tensor`, `IntxUnpackedToInt8Tensor`) and plain tensors.
Metadata records each subclass's type and attributes. No backend-specific
packing — packing for the target backend happens at load time via
`quant.pack_model()`.

## Quantization recipes

Two built-in recipes (see `quantize_and_save.py`):

| Recipe | Description |
|---|---|
| `default` | INT4 min_max linears, INT8 per-axis embedding |
| `sensitive` | INT8 for edge-layer v_proj/down_proj, INT4 hqq elsewhere, INT8 per-axis embedding |

The vision tower position-embedding table is quantized to INT8 per-channel
inside `quantize_and_save.py` (via `quant/pack_vision_cuda.py::quantize_vision_position_table`)
and persisted as int8 + fp32-scale buffers in the prequantized checkpoint;
all other vision params stay in bf16 (~0.4 GB).

## Recommended flow: prequantize once, then export

The recommended flow is two separate steps -- there is no
quantization-during-export path:

1. **Prequantize once.** Run `quantize_and_save.py` on the bf16 HF checkpoint
   to produce a prequantized safetensors checkpoint (~25 min, ~30 GB CPU RAM
   peak). Do this once and reuse the result.
2. **Export from the prequantized checkpoint.** Run `export.py --prequantized
   <dir>` to produce `model.pte` + `model.ptd` (~30 min). Export does NOT
   re-quantize; it simply lowers the already-quantized weights into the
   backend graph.

Eager inference (`inference.py`) and the runner both load the same
prequantized checkpoint. The text decoder weights are stored as torchao
`Int4Tensor` / `IntxUnpackedToInt8Tensor` subclasses; the vision tower is
bf16 + the INT8 PE buffers; pack-time is a no-op (no requant happens on
load).

## Prequantized checkpoint

A vision-enabled prequantized checkpoint (sensitive recipe + vision tower
weights for image+text inference) is available on HuggingFace:

```bash
huggingface-cli download gasoonjia/gemma-4-31B-it-HQQ-INT4 --local-dir gemma-4-31B-it-HQQ-INT4
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
    --model-dir /path/to/gemma-4-31B \
    --output ./gemma4_31b_int4 \
    --quant-recipe sensitive
```

See [Quantization recipes](#quantization-recipes) above for details on each
recipe. Writes `model.safetensors`, `config.json`, and `tokenizer.json` into
`--output`. Vision tower weights are preserved through this flow.

## Export to ExecuTorch

### CUDA

```bash
python examples/models/gemma4_31b/export.py \
    --prequantized ./gemma4_31b-it-HQQ-INT4-vision \
    --output-dir ./gemma4_31b_exports \
    --max-seq-len 4096 \
    --backend cuda
```

For CUDA, the exported `.pte` is multimodal — it bundles four methods
(`embed_text`, `vision_encoder`, unified prefill `forward`, and
`decode_forward`), which the runner stitches together for image+text
inference. Text-only inference just skips the `vision_encoder` call.

### MLX (Apple Silicon)

```bash
python examples/models/gemma4_31b/export.py \
    --prequantized ./gemma4_31b_int4 \
    --output-dir ./gemma4_31b_exports_mlx \
    --max-seq-len 4096 \
    --backend mlx
```

The same quantized checkpoint works for both backends. MLX exports a
single `forward` method with dynamic sequence length and host-side
sampling (text-only; no vision support on MLX yet).

Writes `model.pte` (and optionally `model.ptd`) into `--output-dir`.

## Eager inference

The prompt is automatically wrapped with the Gemma 4 IT chat template.
Pass `--raw-prompt` to skip template wrapping for pre-formatted input.

### Text-only

```bash
python examples/models/gemma4_31b/inference.py \
    --prequantized ./gemma4_31b-it-HQQ-INT4-vision \
    --prompt "Write a short joke about saving RAM." \
    --max-new-tokens 128 \
    --temperature 0.8
```

### Image + text

```bash
python examples/models/gemma4_31b/inference.py \
    --prequantized ./gemma4_31b-it-HQQ-INT4-vision \
    --image-path docs/source/_static/img/et-logo.png \
    --prompt "Describe this image." \
    --max-new-tokens 128 \
    --temperature 0
```

The flow mirrors the C++ runner: patchify → vision tower → splice the
image rows into the embedding tensor at `<image>` placeholder positions →
prefill on `inputs_embeds` → decode.

### GGUF

GGUF files from the community (e.g., Q4_K_M, Q4_K_S) can also be used
directly. Community GGUFs pack the text decoder only, so the vision
tower is auto-loaded from an HF bf16 directory resolved in this order:

  1. `GEMMA4_31B_HF_DIR` environment variable
  2. Well-known default `/home/gasoonjia/models/gemma-4-31B`

If neither resolves to a valid HF safetensors checkpoint, the loader
raises with instructions for setting one up.

```bash
python examples/models/gemma4_31b/inference.py \
    --gguf ./gemma-4-31B-it-Q4_K_S.gguf \
    --tokenizer-path /path/to/tokenizer.json \
    --prompt "Hello"
```

Useful before spending the export+lowering time to confirm the quantized
model produces sensible text.

### GGUF + image

The vision tower is auto-loaded from the resolved HF bf16 directory —
no additional flag required:

```bash
python examples/models/gemma4_31b/inference.py \
    --gguf ./gemma-4-31B-it-Q4_K_S.gguf \
    --tokenizer-path /path/to/gemma-4-31B/tokenizer.json \
    --image-path docs/source/_static/img/et-logo.png \
    --prompt "Describe this image." \
    --max-new-tokens 64 \
    --temperature 0 \
    --no-compile
```

The text decoder runs in Q4_K from the GGUF; the vision tower +
multimodal embedder run in bf16 loaded from the HF safetensors shards.
Override the auto-resolved HF dir with `GEMMA4_31B_HF_DIR=/path/to/dir`.

## Build the runner

```bash
make gemma4_31b-cuda   # Linux — CUDA backend
make gemma4_31b-mlx    # macOS — MLX backend (Apple Silicon)
```

The binary lands at `cmake-out/examples/models/gemma4_31b/gemma4_31b_runner`.

## Run the .pte

The prompt is automatically wrapped with the Gemma 4 IT chat template.
Pass `--raw_prompt` to skip template wrapping for pre-formatted input.

### Text-only

```bash
./cmake-out/examples/models/gemma4_31b/gemma4_31b_runner \
    --model_path  ./gemma4_31b_exports/model.pte \
    --data_path   ./gemma4_31b_exports/aoti_cuda_blob.ptd \
    --tokenizer_path ./gemma4_31b-it-HQQ-INT4-vision/tokenizer.json \
    --prompt "Write a short joke about saving RAM." \
    --max_new_tokens 128 \
    --temperature 0.8
```

### Image + text

```bash
./cmake-out/examples/models/gemma4_31b/gemma4_31b_runner \
    --model_path  ./gemma4_31b_exports/model.pte \
    --data_path   ./gemma4_31b_exports/aoti_cuda_blob.ptd \
    --tokenizer_path ./gemma4_31b-it-HQQ-INT4-vision/tokenizer.json \
    --image_path docs/source/_static/img/et-logo.png \
    --prompt "Describe this image." \
    --max_new_tokens 64 \
    --temperature 0
```

For benchmarking, add `--cuda_graph` to capture the decode method in a CUDA
graph (decode is fully static — `T=1`).
