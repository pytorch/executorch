# Gemma 4 31B-IT

Text-only export of Google's Gemma 4 31B-IT to ExecuTorch with INT4/INT8
weight quantization. Supports CUDA and MLX (Apple Silicon) backends.

For architecture and design notes see [model.md](model.md).

## When to use which script

The full bf16 weights for 31B (~62 GB) often don't fit in available RAM. The
recommended flow is to quantize once and reuse the quantized checkpoint for
both export and eager inference:

| Script | Purpose | Peak memory |
|---|---|---|
| `quantize_and_save.py` | bf16 HF checkpoint → quantized checkpoint (one-time) | ~30 GB CPU |
| `export.py --prequantized <dir>` | quantized checkpoint → `model.pte` + `model.ptd` | ~24 GB CPU + CUDA for packing |
| `export.py --gguf <file> [--backend mlx]` | GGUF file (Q4_K_M, etc.) → `model.pte` + `model.ptd` | ~24 GB CPU |
| `inference.py --prequantized <dir>` | quantized checkpoint → eager generation under `torch.compile` | ~24 GB GPU |
| `inference.py --gguf <file>` | GGUF file (Q4_K_M, etc.) → eager generation | ~24 GB GPU |
| `export.py --model-dir <hf>` | one-shot bf16 → quantize → export (no intermediate file) | ~30 GB CPU + CUDA for packing |

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

### CUDA

```bash
python examples/models/gemma4_31b/export.py \
    --prequantized ./gemma4_31b_int4 \
    --output-dir ./gemma4_31b_exports \
    --max-seq-len 4096 \
    --backend cuda
```

### MLX (Apple Silicon)

```bash
python examples/models/gemma4_31b/export.py \
    --prequantized ./gemma4_31b_int4 \
    --output-dir ./gemma4_31b_exports_mlx \
    --max-seq-len 4096 \
    --backend mlx
```

The same quantized checkpoint works for both backends. MLX exports a single
method with dynamic sequence length and host-side sampling.

Writes `model.pte` (and optionally `model.ptd`) into `--output-dir`.

#### TurboQuant KV cache (long context, CUDA + MLX)

For long-context inference, add `--turboquant` to swap the full-attention
layers' KV cache for a TurboQuant TQ4 cache (4-bit codebook + nibble pack).
This gives ~3.8× cache memory savings on the full-attention layers and lets
you fit context lengths that wouldn't fit in bf16. Sliding-window layers are
unaffected. Supported on both the CUDA and MLX backends.

**Long context requires BOTH flags**: `--turboquant` *and* a larger
`--max-seq-len`. Raising `--max-seq-len` alone keeps a bf16 KV cache, which does
not fit at long context. On CUDA, `--turboquant` is what enables 128k: Gemma4-31B
at `--max-seq-len 131072` runs within ~27 GiB at runtime (fits a 32 GB card).

```bash
# CUDA — 128k context (TQ4 KV)
python examples/models/gemma4_31b/export.py \
    --gguf ./gemma-4-31B-it-Q4_K_M.gguf \
    --output-dir ./gemma4_31b_exports_128k \
    --max-seq-len 131072 \
    --backend cuda \
    --turboquant
```

```bash
# MLX (Apple Silicon)
python examples/models/gemma4_31b/export.py \
    --prequantized ./gemma4_31b_int4 \
    --output-dir ./gemma4_31b_exports_mlx_tq \
    --max-seq-len 65536 \
    --backend mlx \
    --turboquant
```

Use TurboQuant when you need context beyond what bf16 fits; otherwise leave it off.

## Eager inference

The prompt is automatically wrapped with the Gemma 4 IT chat template.
Pass `--raw-prompt` to skip template wrapping for pre-formatted input.

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

## Build the runner and worker

```bash
make gemma4_31b-cuda   # Linux — CUDA backend
make gemma4_31b-mlx    # macOS — MLX backend (Apple Silicon)
```

The binaries land at:

- `cmake-out/examples/models/gemma4_31b/gemma4_31b_runner`
- `cmake-out/examples/models/gemma4_31b/gemma4_31b_worker`

## Run the .pte

The prompt is automatically wrapped with the Gemma 4 IT chat template.
Pass `--raw_prompt` to skip template wrapping for pre-formatted input.

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

## OpenAI-compatible serving harness

The serving path is a test harness for local-agent workflows. Python owns HTTP,
chat templating, request validation, and tool parsing; the C++ worker owns model
loading, prefill/decode, and session state. Use the runner or engine/session API
directly for production integrations.

### CUDA

```bash
python -m executorch.examples.models.gemma4_31b.serve \
    --model-path ./gemma4_31b_exports/model.pte \
    --data-path ./gemma4_31b_exports/aoti_cuda_blob.ptd \
    --tokenizer-path ./gemma4_31b_int4/tokenizer.json \
    --hf-tokenizer ./gemma4_31b_int4 \
    --model-id gemma4_31b \
    --max-context 4096 \
    --max-sessions 4 \
    --host 127.0.0.1 \
    --port 8000
```

### MLX

```bash
python -m executorch.examples.models.gemma4_31b.serve \
    --model-path ./gemma4_31b_exports_mlx/model.pte \
    --tokenizer-path ./gemma4_31b_int4/tokenizer.json \
    --hf-tokenizer ./gemma4_31b_int4 \
    --model-id gemma4_31b \
    --max-context 4096 \
    --max-sessions 4 \
    --host 127.0.0.1 \
    --port 8000
```

Named sessions use one loaded model with isolated mutable state when the backend
supports it. Set `--max-sessions >= 2` and send a stable `session_id` (or one of
the supported affinity headers) to enable separate conversations and warm
append-only resume. One capacity slot is reserved for anonymous requests.

The default parser is Gemma's tool-call format. Use `--tool-parser hermes`,
`--tool-parser qwen`, or `--tool-parser none` if a different prompt/template
emits another format.

### CUDA no-bleed test

The CUDA build also produces `test_gemma4_31b_nobleed`, which validates that
two sessions can interleave prefill/decode on one loaded model without sharing
mutable state:

```bash
GEMMA_MODEL_PATH=gemma4_31b_exports/model.pte \
GEMMA_DATA_PATH=gemma4_31b_exports/aoti_cuda_blob.ptd \
GEMMA_TOKENIZER_PATH=gemma4_31b_int4/tokenizer.json \
  cmake-out/examples/models/gemma4_31b/test_gemma4_31b_nobleed
```
