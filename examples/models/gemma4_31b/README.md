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
| `inference.py --gguf <file>` | GGUF text decoder → eager generation (text-only by default; pair with `--vision-from-hf` for image input) | `--image-path` + `--vision-from-hf` | ~24 GB GPU |
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

The vision tower position-embedding table is additionally quantized to INT8
on the fly (see `quant/pack_vision_cuda.py`); all other vision params stay
in bf16 (~0.4 GB).

## Prequantized checkpoint

A vision-enabled prequantized checkpoint (sensitive recipe + vision tower
weights for image+text inference) is available on HuggingFace:

```bash
huggingface-cli download gasoonjia/gemma-4-31B-it-HQQ-INT4-vision --local-dir gemma-4-31B-it-HQQ-INT4-vision
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
directly:

```bash
python examples/models/gemma4_31b/inference.py \
    --gguf ./gemma-4-31B-it-Q4_K_S.gguf \
    --tokenizer-path /path/to/tokenizer.json \
    --prompt "Hello"
```

Useful before spending the export+lowering time to confirm the quantized
model produces sensible text.

### GGUF + image (vision tower from HF bf16)

Community GGUFs pack the text decoder only — the vision tower must come
from an HF bf16 checkpoint. Pair `--gguf` with `--vision-from-hf <hf_dir>`
to enable image input on the GGUF path:

```bash
python examples/models/gemma4_31b/inference.py \
    --gguf ./gemma-4-31B-it-Q4_K_S.gguf \
    --vision-from-hf /path/to/gemma-4-31B \
    --tokenizer-path /path/to/gemma-4-31B/tokenizer.json \
    --image-path docs/source/_static/img/et-logo.png \
    --prompt "Describe this image." \
    --max-new-tokens 64 \
    --temperature 0 \
    --no-compile
```

The text decoder runs in Q4_K from the GGUF; the vision tower +
multimodal embedder run in bf16 loaded from the HF safetensors shards.

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
