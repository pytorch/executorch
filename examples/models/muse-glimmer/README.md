# Muse Glimmer 30B

ExecuTorch export and runtime support for Muse Glimmer 30B, including text,
vision, DFlash speculative decoding, and OpenAI-compatible serving. Quantized
GGUF checkpoints are the recommended starting point and lower directly to the
selected ExecuTorch backend.

## Supported backends

| Backend | Host | Exported artifacts |
|---|---|---|
| CUDA | Linux or Windows | `model.pte` plus `aoti_cuda_blob.ptd` |
| MLX | macOS on Apple silicon | Self-contained `model.pte` |

CPU export is not supported. For general environment setup, follow the
[ExecuTorch README](../../../README.md) and
[build-from-source guide](../../../docs/source/using-executorch-building-from-source.md).

## Model assets

The GGUF checkpoints below are the inputs to [Export](#export). To run a
published export instead, skip to
[Prebuilt PTE artifacts](#prebuilt-pte-artifacts); that download is
self-contained and needs nothing else from this section.

Download the GGUF checkpoints and Hugging Face tokenizer metadata from the
[`meta-models/Muse-Glimmer-30B-GGUF`](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)
and [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B),
respectively. Run from the ExecuTorch repository root:

```bash
GGUF_REPO=meta-models/Muse-Glimmer-30B-GGUF
HF_REPO=meta-models/Muse-Glimmer-30B

hf download "$GGUF_REPO" \
  --include '*.gguf' \
  --exclude '*[Bb][Ff]16*.gguf' \
  --local-dir assets/quant

hf download "$HF_REPO" \
  tokenizer.json tokenizer_config.json chat_template.jinja \
  --local-dir assets/hf
```

The canonical `chat_template.jinja` must remain beside the tokenizer metadata;
the serving path uses it to render prompts and tools.

### GGUF checkpoints

| File | Use |
|---|---|
| `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` | Recommended target checkpoint |
| `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf` | Dynamic K-quant target checkpoint |
| `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` | Optional vision projector |
| `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` | Optional DFlash draft checkpoint |

Set paths once for the commands below:

```bash
TARGET="$(find assets/quant -type f -name 'Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf' -print -quit)"
DRAFT="$(find assets/quant -type f -name 'dflash-Muse-Glimmer-30B-Q4_K_M.gguf' -print -quit)"
MMPROJ="$(find assets/quant -type f -name 'mmproj-Muse-Glimmer-30B-Q4_K_M.gguf' -print -quit)"
BACKEND=cuda  # use mlx on macOS
```

### Prebuilt PTE artifacts

Ready-to-run exports for Apple silicon and NVIDIA GPUs are published in
[`meta-models/Muse-Glimmer-30B-ExecuTorch-PTE`](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE).
A prebuilt export needs neither the GGUF checkpoints nor the [Export](#export)
step, and the repository serves the tokenizer metadata and chat template at its
root, so it is the only download required.

Each subdirectory is one export, named after its quantization, context length,
modalities, decoding mode (`solo` or `dflash`), and target backend (`metal` for
Apple silicon, `sm80+ptx` for NVIDIA), for example
`muse-glimmer-k-quant-17G-128K-text-solo-metal`. Its artifacts repeat that
directory name rather than the `model.pte` and `aoti_cuda_blob.ptd` filenames a
local export writes:

| Artifact | Present |
|---|---|
| `<subdirectory>.pte` | Always |
| `<subdirectory>.ptd` | `sm80+ptx` only, and it holds the weights, so a CUDA export needs both files |
| `pos_embed.bin` | `text-image` only |

Browse the repository, pick the subdirectory that matches your setup, and
download just that one alongside the shared tokenizer metadata:

```bash
PTE_REPO=meta-models/Muse-Glimmer-30B-ExecuTorch-PTE
EXPORT_DIR=muse-glimmer-k-quant-17G-128K-text-solo-metal

hf download "$PTE_REPO" \
  --include "$EXPORT_DIR/*" \
  --include tokenizer.json \
  --include tokenizer_config.json \
  --include chat_template.jinja \
  --local-dir exports
```

`--include` takes one pattern per flag, and adding positional filenames makes
`hf` ignore every `--include`. The repository holds many exports, so avoid
downloading it in full; `--dry-run` reports the file list and total size before
the transfer starts.

With a prebuilt export, skip the [Export](#export) section and use
`exports/$EXPORT_DIR` as the artifact directory and `exports` as the tokenizer
directory in the sections below.

## Export

Run exports from the ExecuTorch repository root.

Target model:

```bash
python -m executorch.examples.models.muse_glimmer.export.export_solo \
  --gguf "$TARGET" \
  --backend "$BACKEND" \
  --output-dir exports/solo
```

DFlash:

```bash
python -m executorch.examples.models.muse_glimmer.export.export_dflash \
  --target-gguf "$TARGET" \
  --draft-gguf "$DRAFT" \
  --backend "$BACKEND" \
  --output-dir exports/dflash
```

Add `--mmproj "$MMPROJ"` to either command to include the vision encoder. A
vision export also writes `pos_embed.bin` beside `model.pte`.

The quick download excludes BF16 GGUFs, but they remain supported inputs. For a
custom local quantization, `export_solo` also accepts a consolidated BF16
checkpoint with `--checkpoint-dir` and a recipe selected by `--quant-recipe`.
Prequantized and MLX-affine checkpoints are available through `--prequantized`
and `--mlx`. Run the export modules with `--help` for the supported recipes and
advanced inputs.

### Exported methods

| Export | Backend | Methods |
|---|---|---|
| Target | CUDA | `embed_text`, `forward_from_embeddings`, `decode_from_embedding` |
| Target | MLX | `embed_text`, `forward_from_embeddings` |
| DFlash | CUDA | `target_forward_from_embeddings`, `target_prefill_from_embeddings`, `embed_text`, `draft_forward`, `draft_prefill`, `dflash_sample_tokens`, `dflash_verify_speculative` |
| DFlash | MLX | `target_forward_from_embeddings`, `embed_text`, `draft_forward` |

Vision adds `vision_encoder` to each method set. CUDA DFlash samples draft
proposals, verifies them, and samples corrections through its two
`dflash_*` methods, so vocabulary-wide logits and probabilities stay on the
device; MLX DFlash samples on the host.

CUDA DFlash exports support the draft model's block size up to 16 positions.
The runtime keeps the draft hidden-state input at that exported width and copies
only committed target hidden rows back to the host after verification. Runner
defaults remain three proposals in a four-position block. To use all 15 proposals
with an artifact exported for a 16-position block, pass `--n_draft=15` and
`--block_length=16` to `dflash_runner`.

## Build the runners

From the repo root, the make targets build ExecuTorch with the backend and then
the runners:

```bash
# CUDA
make muse-glimmer-cuda

# MLX
make muse-glimmer-mlx
```

When ExecuTorch is already built for that backend, the model's CMake workflow
preset rebuilds just the runners:

```bash
# CUDA
(cd examples/models/muse-glimmer && \
  cmake --workflow --preset muse-glimmer-cuda)

# MLX
(cd examples/models/muse-glimmer && \
  cmake --workflow --preset muse-glimmer-mlx)
```

The binaries are written to `cmake-out/examples/models/muse-glimmer/`:

- `solo_runner`
- `dflash_runner`
- `muse_glimmer_worker`

Run a binary with `--help` for its model-specific arguments.

## Serve

The following instructions use the regular worker built above. For off-graph
exports, use [the batching server](#start-the-batching-server) below instead.

Install the shared server dependencies:

```bash
pip install -r /path/to/executorch/examples/llm_server/python/requirements.txt
```

Serve a CUDA target export, replacing the example paths with your local paths:

```bash
python -m executorch.examples.models.muse_glimmer.serving.serve \
  --model-path /path/to/exports/solo/model.pte \
  --data-path /path/to/exports/solo/aoti_cuda_blob.ptd \
  --tokenizer-path /path/to/assets/hf/tokenizer.json \
  --hf-tokenizer /path/to/assets/hf \
  --worker-bin /path/to/executorch/cmake-out/examples/models/muse-glimmer/muse_glimmer_worker \
  --model-id muse-glimmer-30B \
  --tool-parser atem \
  --host 127.0.0.1 \
  --port 8000
```

For MLX, use the MLX export and omit `--data-path`. For DFlash, use the matching
DFlash export paths; the server detects the exported method contract. For
vision, also pass `--pos-embed-path /path/to/exports/solo/pos_embed.bin`, using
the matching export directory.

Serving a prebuilt export instead follows the same shape, with both artifacts
named after their subdirectory and the tokenizer metadata taken from the
download root:

```bash
EXPORT_DIR=muse-glimmer-k-quant-17G-128K-text-solo-sm80+ptx

python -m executorch.examples.models.muse_glimmer.serving.serve \
  --model-path "exports/$EXPORT_DIR/$EXPORT_DIR.pte" \
  --data-path "exports/$EXPORT_DIR/$EXPORT_DIR.ptd" \
  --tokenizer-path exports/tokenizer.json \
  --hf-tokenizer exports \
  --worker-bin cmake-out/examples/models/muse-glimmer/muse_glimmer_worker \
  --model-id muse-glimmer-30B \
  --tool-parser atem \
  --host 127.0.0.1 \
  --port 8000
```

`--dflash-block-length` defaults to 4, the largest block the CUDA export
supports. The MLX export carries the drafter's native block size of 16, so pass
`--dflash-block-length 0` there to select the exported maximum; `0` for
`--dflash-n-draft` likewise resolves to `block_length - 1`. On `sm80+ptx`,
`--dflash-n-draft` must stay at 3 or below.

Smoke test:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"muse-glimmer-30B","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}'
```

Tool calling is supported with `--tool-parser atem`. The canonical Hugging Face
template renders tool definitions, and the server converts Muse Glimmer 30B ATEM
output into OpenAI-compatible `tool_calls`.

## Off-graph KV cache and batching

The off-graph path keeps KV cache state in the runtime rather than inside the
exported model graph. It supports packed multi-session batching, runtime-managed
cache growth, and optional prefix snapshots for reuse across requests.
Batching supports both text-only and vision-enabled exports, with or without
DFlash.

It uses separate batching runners and workers. Regular runner artifacts and
workers are not interchangeable with batching artifacts and workers.

The commands below use MLX as an example. Select the backend matching your
hardware, export, and worker build.

### Export

Export with `--use-offgraph-kv-cache`. For example, using MLX and the checkpoint
variables from [Model assets](#model-assets), export a text-only target:

```bash
python -m executorch.examples.models.muse_glimmer.export.export_solo \
  --backend mlx \
  --gguf "$TARGET" \
  --use-offgraph-kv-cache \
  --activation-dtype float16 \
  --max-prefill-chunk 512 \
  --output-dir /path/to/exports/mg-offgraph
```

Or export the target and drafter for DFlash:

```bash
python -m executorch.examples.models.muse_glimmer.export.export_dflash \
  --backend mlx \
  --target-gguf "$TARGET" \
  --draft-gguf "$DRAFT" \
  --use-offgraph-kv-cache \
  --activation-dtype float16 \
  --output-dir /path/to/exports/mg-offgraph-dflash
```

For CUDA exports, use `--backend cuda` and `--activation-dtype bfloat16` instead.

To enable image input, add `--mmproj "$MMPROJ"` to either export command above.
This includes the vision encoder and writes its position-embedding data alongside
the model. The resulting export can serve both text-only and image requests;
a separate text-only export is not required.

The server's `--max-context` sets the per-session runtime limit and must not
exceed the exported limit. Runtime cache allocation is separate from that limit.
For MLX, off-graph exports use the checkpoint's native context limit rather than
`--max-seq-len`, and KV storage grows as needed instead of allocating the full
context up front.

### Set up the batching runtime

After the general [environment setup](#supported-backends), use the Make target
for your backend to build and install ExecuTorch and build the batching runner
and worker. For MLX:

```bash
cd /path/to/executorch
make muse-glimmer-batching-mlx
pip install -r /path/to/executorch/examples/llm_server/python/requirements.txt
```

The Make target builds the native dependencies and both batching executables;
it does not export a model or start the server. Use a batching build target,
not a regular runner target.

### Start the batching server

Replace the absolute example paths below with your checkout, export, and
tokenizer locations. Select the backend matching the export and worker build.
This MLX example uses four concurrent sessions, a 128K (131072-token) context
limit, and up to 32 prefix snapshots:

```bash
python -m executorch.examples.models.muse_glimmer.serving.serve_batching \
  --worker-bin /path/to/executorch/cmake-out/examples/models/muse-glimmer-batching/muse_glimmer_batching_worker \
  --model-path /path/to/exports/mg-offgraph/model.pte \
  --tokenizer-path /path/to/assets/hf/tokenizer.json \
  --hf-tokenizer /path/to/assets/hf \
  --backend mlx \
  --model-id muse-glimmer-30B \
  --max-context 131072 \
  --max-sessions 4 \
  --max-decode-sequences 4 \
  --max-inflight-requests 4 \
  --prefix-cache-entries 32 \
  --tool-parser atem \
  --host 127.0.0.1 \
  --port 8000
```

For DFlash, use `/path/to/exports/mg-offgraph-dflash/model.pte` and add
`--dflash --dflash-n-draft 3` to the launch command. `--dflash-n-draft` is a
runtime server option controlling the number of proposals per verification step
(N). Set it when starting the server; changing N does not require re-exporting.
N must be smaller than the exported drafter block size (D). Omitting the option,
or setting it to zero, uses N = D - 1.

On CUDA, D is fixed at export time from the drafter checkpoint configuration and
cannot be changed at runtime. The exporter currently has no separate block-size
override option. To use a different D, export a new artifact with the corresponding
drafter configuration. Changing N does not change D: a D = 16 export still
evaluates 16 drafter rows per session when N = 3, and its default is N = 15.

Choose concurrency and prefix-cache limits to fit available memory. Prefix
snapshots are disabled by default; omit `--prefix-cache-entries` to disable them.
If the export stores weights separately, also pass `--data-path` with its data
file.

For a vision-enabled export, add
`--pos-embed-path /path/to/exports/mg-offgraph/pos_embed.bin` to the server command,
using the matching export directory. Send images as OpenAI `image_url` content
parts containing inline base64 JPEG or PNG data URLs. The batching server accepts
one image across each submitted conversation; remote image URLs and local image
paths are not supported.

The endpoint is `http://127.0.0.1:8000/v1/chat/completions`; the smoke request in
[Serve](#serve) and the [Pi configuration](#use-from-pi) below work with this
server too. See the
[batching serving guide](BATCHING_SERVING.md) for image support, session behavior,
and detailed limits.

## Use from Pi

This configuration works with both the regular server and the off-graph
batching server. Batching and DFlash are server-side choices; Pi uses the same
OpenAI-compatible endpoint for either path. Enable `--tool-parser atem` on the
server for tool calling.

Point Pi at the server via `/path/to/home/.pi/agent/models.json`, replacing
`/path/to/home` with your home directory:

```json
{
  "providers": {
    "muse-glimmer-local": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "x",
      "models": [
        {
          "id": "muse-glimmer-30B",
          "reasoning": true,
          "contextWindow": 131072,
          "maxTokens": 32768,
          "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false,
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": { "return_reasoning": true },
            "sendSessionAffinityHeaders": true
          }
        }
      ]
    }
  }
}
```

```bash
pi --provider muse-glimmer-local \
  --model muse-glimmer-30B \
  --thinking high \
  --tools read,bash,edit,write
```

The model id must match `--model-id`. The `compat` entries:

- `supportsDeveloperRole` — the template renders no `developer` turn, so pi's
  system prompt is otherwise dropped silently.
- `supportsReasoningEffort` — the server rejects `reasoning_effort` with a 400.
- `return_reasoning` — defaults to `true`, returning the `to=self` channel as
  `reasoning_content` (or `delta.reasoning_content` when streaming). Set
  `"chatTemplateKwargs": { "return_reasoning": false }` to omit it. This only
  controls the response; the model still computes reasoning.
- `sendSessionAffinityHeaders` — optional, for per-conversation sessions; needs
  `--max-sessions` above 1.

Set `contextWindow` to the export's context length (`128K` is 131072) and pass
the same value as `--max-context`. Add `"input": ["text", "image"]` for a vision
export.

For direct HTTP requests, the opt-out is
`"chat_template_kwargs": { "return_reasoning": false }`. The flag must be a JSON
boolean; strings, numbers, and `null` return a 400 error.
