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
core libraries and the binaries.

```bash
make qwen3_5_moe-cuda
```

This builds ExecuTorch with CUDA backend support, then the runner binary
at `cmake-out/examples/models/qwen3_5_moe/qwen3_5_moe_runner` and the
serving worker at `cmake-out/examples/models/qwen3_5_moe/qwen3_5_moe_worker`
(see [Serving](#serving-openai-compatible)).

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
| `--prompt_file` | (none) | Path to a file with the prompt (overrides `--prompt`) |
| `--temperature` | `0.8` | Sampling temperature (0 = greedy) |
| `--max_new_tokens` | `128` | Maximum tokens to generate |
| `--warmup` | `0` | Warmup iterations to discard before timing (one model load; the session is reset between iterations) |
| `--num_iters` | `1` | Timed iterations to average, after warmup |

## Serving (OpenAI-compatible)

Run an OpenAI-compatible HTTP server so an agent harness (pi, opencode, …) can
use the model for local tool-use. Point your client at `http://<host>:<port>/v1`.

The CUDA build produces the runner **and** the serving worker:

```bash
make qwen3_5_moe-cuda
```

Launch (the `LD_LIBRARY_PATH` shim is forwarded to the worker for the CUDA blob):

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
  python -m executorch.examples.models.qwen3_5_moe.serve \
    --model-path  qwen35_moe_exports/model.pte \
    --data-path   qwen35_moe_exports/aoti_cuda_blob.ptd \
    --tokenizer-path ~/models/Qwen3.5-35B-A3B/tokenizer.json \
    --hf-tokenizer   ~/models/Qwen3.5-35B-A3B \
    --model-id qwen3.5-moe --no-think --max-sessions 4
```

`--max-sessions >= 2` is required for named sessions and warm resume; the default
`1` is scratch-only (one slot is reserved for anonymous requests).

### Architecture (process isolation)

Two processes, one model load:

```
serve.py            (control plane: FastAPI/asyncio, OpenAI protocol, chat
                     templating, tool parsing, validation — NO CUDA, NO pybind)
   │  JSONL over stdin/stdout
   ▼
qwen3_5_moe_worker  (C++ binary: one Qwen35MoEEngine, many isolated sessions,
                     synchronous loop — the CUDA model; NO asyncio server)
```

The model runs in a **separate worker process** because executing the AOTI CUDA
model inside a live asyncio server process segfaults in the int4 matmul
(reproducible, and isolated by elimination to the asyncio-loop × CUDA
interaction). The worker runs the model like the CLI — a plain synchronous loop —
which is reliable. The control plane only does blocking pipe I/O (no CUDA), which
is safe under asyncio.

### Serve Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Path to exported `.pte` model |
| `--data-path` | (none) | Path to `.ptd` delegate data file (required for CUDA) |
| `--tokenizer-path` | (required) | Path to HuggingFace `tokenizer.json` |
| `--hf-tokenizer` | (required) | HF tokenizer id/dir for the chat template + encoding |
| `--model-id` | `qwen3.5-moe` | Model id reported on `/v1/models` |
| `--host` / `--port` | `127.0.0.1` / `8000` | Bind address |
| `--max-context` | (none) | Reject prompts that exceed it with 400 |
| `--no-think` | off | Default reasoning off (`enable_thinking=False`) |
| `--max-sessions` | `1` | Isolated sessions on one weight load (see Sessions) |
| `--warm-resume` / `--no-warm-resume` | on | Reuse a session's KV across turns (see Sessions) |

### Sessions

One worker loads the weights once (~18 GB) and hosts multiple **isolated**
sessions on that single allocation, each with its own KV/recurrent state. Set
`--max-sessions N` (clamped to 1 if the backend hosts a single session); one slot
is reserved for anonymous requests, so up to `N - 1` named `session_id`s are
addressable.

Route a request to a persistent session with the `session_id` body field or, as
aliases, the `X-ExecuTorch-Session-ID` / `session_id` / `x-session-affinity`
headers (body wins, then that header order). The header aliases let a client that
emits a stable per-conversation affinity id route per conversation (for pi, set
`compat.sendSessionAffinityHeaders: true` in models.json). Requests without any
share a transient scratch session.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-moe","session_id":"alice",
       "messages":[{"role":"user","content":"hi"}]}'

curl -X POST   http://127.0.0.1:8000/v1/sessions/alice/reset  # clear context, keep the slot
curl -X DELETE http://127.0.0.1:8000/v1/sessions/alice        # free context + slot (VRAM)
```

Admission is up front: an explicit `session_id` on a single-session server
returns **400** (`unsupported_session`); past capacity it returns **429**
(`capacity_exhausted`) before any response bytes.

**Warm append-only resume** (on by default): when a named session's next request
is an exact-token extension of its resident context (e.g. the same conversation
plus a new turn), the worker prefills **only the new suffix** instead of
re-prefilling the whole prompt — continuing the KV/recurrent state in place. The
check is exact-token (never re-tokenized text), so it is always correct: anything
that can't be proven an exact extension (token mismatch, a stop-string trim, a
prior error) falls back to a full reset + prefill. This is **per-session** warm
append-only resume, **not** global prefix caching: there is no cross-session
prefix sharing, so a system prompt common to two different `session_id`s is
prefilled independently for each (unlike vLLM/llama.cpp global prefix reuse).
Each `done` event reports
`reused_prompt_tokens`, `prefilled_prompt_tokens`, and `session_reset_reason`
(`new`/`exact_prefix`/`dirty`/`mismatch`/`equal`) for measuring the hit rate.
`--no-warm-resume` forces a full prefill every request (for A/B comparison).

**Tool-call turns** also warm-resume: an assistant turn re-rendered from its
parsed tool call rarely re-tokenizes to the tokens the model generated, so the
server replays the exact generated token ids for prior turns to keep the resident
state an exact prefix (tool *results* stay text). A mismatch still falls back to a
full prefill.

This is **isolation + warm resume, not concurrency**: execution is still
synchronous (one in-flight request; `--num-runners > 1` is rejected since more
workers would duplicate the weights). Fair interleaving across in-flight requests
is a follow-up.

### Other limitations

- Supports the chat-completions contract of the generic server; `top_p != 1`,
  `seed`, `top_k`, `logprobs`, etc. are rejected (only temperature is plumbed).

## Troubleshooting

- **Runner exits silently right after `Loading methods...`**: the AOTI CUDA blob
  is compiled with the conda toolchain's `libstdc++`, which is newer than the
  system one (it needs e.g. `GLIBCXX_3.4.34`). Prepend the conda lib dir so the
  runner loads the matching `libstdc++`:

  ```bash
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
    cmake-out/examples/models/qwen3_5_moe/qwen3_5_moe_runner ...
  ```
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
