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
| DFlash | CUDA | `target_forward_from_embeddings`, `target_prefill_from_embeddings`, `embed_text`, `draft_forward`, `draft_prefill` |
| DFlash | MLX | `target_forward_from_embeddings`, `embed_text`, `draft_forward` |

Vision adds `vision_encoder` to each method set.

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
- `return_reasoning` — returns the `to=self` channel as `reasoning_content`.
- `sendSessionAffinityHeaders` — optional, for per-conversation sessions; needs
  `--max-sessions` above 1.

Set `contextWindow` to the export's context length (`128K` is 131072) and pass
the same value as `--max-context`. Add `"input": ["text", "image"]` for a vision
export.
