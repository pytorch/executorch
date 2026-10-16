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
Each subdirectory is one export, named after its quantization, context length,
modalities, decoding mode (`solo` or `dflash`), and target hardware, for example
`muse_glimmer_k_quant_17G_128K_text_solo_metal`. Browse the repository, pick the
subdirectory that matches your setup, and download only that one:

```bash
EXPORT_DIR=<subdirectory from the repository>

hf download meta-models/Muse-Glimmer-30B-ExecuTorch-PTE \
  --include "$EXPORT_DIR/*" \
  --local-dir exports
```

The repository holds many exports, so avoid downloading it in full.

With a prebuilt export, skip the [Export](#export) section and use
`exports/$EXPORT_DIR` as the artifact directory in the sections below.

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

## Off-graph KV cache and batching

The off-graph path keeps KV cache state in the runtime rather than inside the
exported model graph. It supports packed multi-session batching, runtime-managed
cache growth, and optional prefix snapshots for reuse across requests. The
batching server uses a shared serving interface with backend-specific cache and
model execution implementations, with optional DFlash speculative decoding.
It uses separate batching runners and requires an off-graph export; legacy
artifacts and workers are not interchangeable with it.

MLX is currently supported; CUDA support is planned. The commands below show
MLX as the available backend example, not a restriction of the off-graph design.

### Export

Select a supported backend
example, using MLX and the checkpoint variables from [Model assets](#model-assets),
export a text-only target:

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

The server's `--max-context` sets the per-session runtime limit and must not
exceed the exported limit. Runtime cache allocation is separate from that limit.
For MLX, off-graph exports use the checkpoint's native context limit rather than
`--max-seq-len`, and KV storage grows as needed instead of allocating the full
context up front.

### Set up the batching runtime

After the general [environment setup](#supported-backends), use the Make target
for your backend to build and install ExecuTorch and build the batching runner
and worker. The current MLX setup command is:

```bash
cd /path/to/executorch
make muse-glimmer-batching-mlx
pip install -r /path/to/executorch/examples/llm_server/python/requirements.txt
```

The Make target builds the native dependencies and both batching executables;
it does not export a model or start the server. CUDA will require its matching
batching build target when support is available; the legacy CUDA runner target
is not a substitute.

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
`--dflash --dflash-n-draft 3` to the launch command. The artifact must support
at least three draft tokens (a block size of four). Omitting `--dflash-n-draft`,
or setting it to zero, uses the exported maximum minus one. Draft proposals are
always argmax; request sampling parameters apply to the target distribution.

Choose concurrency and prefix-cache limits to fit available memory. Prefix
snapshots are disabled by default; omit `--prefix-cache-entries` to disable them.
If the export stores weights separately, also pass `--data-path` with its data
file. For vision, export with `--mmproj "$MMPROJ"` and pass
`--pos-embed-path /path/to/exports/mg-offgraph/pos_embed.bin`, using the matching
export directory.

The endpoint is `http://127.0.0.1:8000/v1/chat/completions`; the smoke request and
pi configuration below work with this server too. See the
[batching serving guide](BATCHING_SERVING.md) for image support, session behavior,
and detailed limits.

## Serve

The following instructions use the legacy worker. For off-graph exports, use
[the batching server](#start-the-batching-server) above instead.

Install the shared server dependencies:

```bash
pip install -r examples/llm_server/python/requirements.txt
```

Serve a CUDA target export:

```bash
python -m executorch.examples.models.muse_glimmer.serving.serve \
  --model-path exports/solo/model.pte \
  --data-path exports/solo/aoti_cuda_blob.ptd \
  --tokenizer-path assets/hf/tokenizer.json \
  --hf-tokenizer assets/hf \
  --worker-bin cmake-out/examples/models/muse-glimmer/muse_glimmer_worker \
  --model-id muse-glimmer-30B \
  --tool-parser atem \
  --host 127.0.0.1 \
  --port 8000
```

For MLX, use the MLX `model.pte` and omit `--data-path`. For DFlash, replace
`exports/solo` with `exports/dflash` in both artifact paths; the server detects
the exported method contract. For vision, also pass
`--pos-embed-path <export-dir>/pos_embed.bin`.

Smoke test:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"muse-glimmer-30B","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}'
```

Tool calling is supported with `--tool-parser atem`. The canonical Hugging Face
template renders tool definitions, and the server converts Muse Glimmer 30B ATEM
output into OpenAI-compatible `tool_calls`. See
[`serving/tool_parsers/atem.py`](serving/tool_parsers/atem.py) and
[`tests/test_atem_tool_parser.py`](tests/test_atem_tool_parser.py).

### Use from pi

Point pi at the server via `~/.pi/agent/models.json`:

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
