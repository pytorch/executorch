# Muse Glimmer Batching Serving

The batching runner and multiplexed worker support MLX autoregressive generation
with text only or text plus one image across the submitted conversation. They are
opt-in entry points; the legacy runners and export defaults remain unchanged.
Batching CUDA and DFlash execution are not supported.

## Export

Export with the off-graph cache ABI required by the batching runner. Include the
vision projector for image support. Replace the placeholder paths with your local
checkpoint and output paths.

```sh
python -m executorch.examples.models.muse_glimmer.export.export_solo \
  --backend mlx \
  --gguf /path/to/models/mg/model.gguf \
  --mmproj /path/to/models/mg/projector.gguf \
  --use-offgraph-kv-cache \
  --activation-dtype float16 \
  --max-prefill-chunk 512 \
  --max-vision-patches 4096 \
  --output-dir /path/to/models/mg-batching
```

This artifact supports packed batching execution and is not interchangeable with
an artifact exported for the legacy runners.

Off-graph MLX export uses the checkpoint's native context limit and ignores
`--max-seq-len`, which remains a legacy export override. Missing or invalid native
context metadata is an error. The three relevant limits are independent:

- The checkpoint's native context limit bounds token positions.
- `--max-prefill-chunk` bounds positions processed per model forward.
- The server's `--max-context` controls the runtime context/cache budget and may be
  lower than the exported limit. Exporting the full context does not allocate that
  full KV capacity at runtime.

## Build

On Apple Silicon, run the Make target from the ExecuTorch checkout:

```sh
cd /path/to/executorch
make muse-glimmer-batching-mlx
```

This builds and installs the MLX dependencies, then builds
`muse_glimmer_batching_runner` and `muse_glimmer_batching_worker` in
`/path/to/executorch/cmake-out/examples/models/muse-glimmer-batching`. It does not
export a model or run tests. Use `make muse-glimmer-mlx` for the legacy runners.

### Custom CMake Builds

The Make target handles the normal configuration. For a custom build, enable
`MUSE_GLIMMER_BUILD_BATCHING` in the MG example and use an MLX-enabled ExecuTorch
installation with `EXECUTORCH_BUILD_EXTENSION_LLM` and
`EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER` enabled. Set `EXECUTORCH_BUILD_MLX=ON` and
`EXECUTORCH_BUILD_CUDA=OFF` in the example configuration as well.

Optional overrides are `MUSE_GLIMMER_STB_INCLUDE_DIR` for local stb headers and
`MLX_METALLIB_PATH` when the installed package does not supply the metallib.
Preparation/materialization tests can be enabled with
`MUSE_GLIMMER_BUILD_BATCHING_TESTS=ON`, built as `muse_glimmer_batching_test`, and
run with CTest's `muse_glimmer_batching` test. Both MG build options default to
`OFF`; the tests can be built without the batching executor but still require the
MLX configuration.

## Serve

Use the worker built by the Make target and the exported model artifacts:

```sh
python -m executorch.examples.models.muse_glimmer.serving.serve_batching \
  --worker-bin /path/to/executorch/cmake-out/examples/models/muse-glimmer-batching/muse_glimmer_batching_worker \
  --model-path /path/to/models/mg-batching/model.pte \
  --data-path /path/to/models/mg-batching/model.ptd \
  --pos-embed-path /path/to/models/mg-batching/pos_embed.bin \
  --tokenizer-path /path/to/models/mg/tokenizer.json \
  --hf-tokenizer /path/to/models/mg \
  --backend mlx \
  --max-context 16384
```

Omit `--data-path` when weights are embedded in the program. For text-only
artifacts, omit `--pos-embed-path`. The launcher uses one multiplexed worker
process and the matching Hugging Face chat template. It does not fall back to a
legacy worker or DFlash.

Send `"stream": true` for streaming responses. The MG parser removes model
framing and emits `reasoning_content` and `content` incrementally. Optional ATEM
tool parsing is enabled with `--tool-parser atem`; tool-enabled responses
currently remain buffered.

Image requests use the OpenAI `image_url` content part with an inline base64 JPEG
or PNG data URL. Remote URLs, file paths, `detail`, and more than one image across
the entire history are rejected. Resubmit the image inline when it is part of the
submitted history; server-side image references are not supported.

### Limits

- `--max-sessions`, `--max-inflight-requests`, and `--max-decode-sequences` default
  to 4. They control session capacity, request admission, and decode batch size
  separately. Set all three according to available memory and desired concurrency.
- `--max-inflight-requests` accepts positive signed-32-bit integers without a
  build-time concurrency ceiling. The worker advertises its capacity to the client.
- `--max-context` must be greater than 1, fit a signed 32-bit integer, and be
  supported by the exported model.
- `--max-image-bytes` is 1 byte through 20 MiB (default 20 MiB), measured on the
  compressed image after base64 decoding, not on decoded pixels.
- `--max-request-bytes` is positive and at most 32 MiB (default 32 MiB), including
  JSONL framing and the newline. It must be at least
  `4 * ceil(max_image_bytes / 3) + 1024`; lower both limits together when needed.
- Output frames have a separate fixed 1 MiB limit. BOS/EOS IDs must fit unsigned
  64-bit integers.

The base64/framing check does not guarantee that an arbitrarily long text history
fits. Whole-frame, decoded-pixel, vision-patch, and model context limits still
apply. Text prefix snapshots are opt-in via `--prefix-cache-entries` (default 0).

## Session Behavior

- Text-only prompts use the shared batching token-history and prefix-reuse path.
- Image prompts count image rows as decoder positions. They replace the session
  context and bypass prompt-prefix caching instead of appending image bytes to a
  warm text prefix.
- A subsequent text-only prompt replays its context after image execution. KV
  cache state remains active within each generation.
- Metadata, options, and prompt preparation are validated before context
  replacement. A later engine or vision failure does not restore replaced state
  and follows the shared batching failure contract.

## Implementation

`MuseGlimmerMLXExecutor` implements `batching::Executor` directly and owns model
execution, MLX cache/sequence state, packing, sampling, and embedding
materialization. The worker reuses the shared batching, serving, and multiplexed
transport infrastructure.

CPU prompt preparation runs after admission on the control thread. Vision
encoding, text embeddings, and decoder calls run on the engine thread. Image
embeddings are computed once per prepared input and can be sliced across model
forwards.
