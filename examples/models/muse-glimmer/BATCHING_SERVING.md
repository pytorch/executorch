# Batching Solo Serving

The batching solo runner and multiplexed worker are additive, opt-in entry points
for MLX autoregressive generation with text only or text plus one image across
the entire submitted history. Existing solo, DFlash, legacy worker, launcher,
and export defaults remain unchanged. Batching CUDA and DFlash execution are not
implemented; there is no fallback to either.

## Executor Ownership

`MuseGlimmerMLXExecutor` implements `batching::Executor` directly. It owns its
module execution, MLX cache/sequence state, packing, sampling, and image
embedding materialization. It is not a `ModuleExecutor` subclass or wrapper and
does not add MG-specific hooks to the generic `ModuleExecutor`. The backend
factory API remains unchanged; runners use the shared batching and serving
interfaces. A future DFlash executor would be a separate implementation, not a
feature of this path.

The batching runtime links the batching, cache, sampler, module, tensor, and
serving targets directly, without `extension_llm_batching_module`. Transport
reuses `multiplexed_worker` if that target already exists; otherwise an MG-local
library compiles the shared transport source and publishes its JSON include
headers. It does not import the shared worker example/test CMake tree.

## Export

Opt in to the batching off-graph cache ABI. Include the vision projector for image
support. All paths below are portable placeholders to replace with local paths;
these commands are for the user to run, not evidence of completed validation.

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

The default export remains compatible with the legacy runners. The opt-in
artifact instead provides per-token positions, selected logits, and neutral
cache geometry for packed batching execution. It requires a cache-aware batching
runner; it is not a replacement artifact for the legacy execution path.

Off-graph MLX export uses the checkpoint's native context limit and ignores
`--max-seq-len`, which remains a legacy export override. The native limit comes
from the GGUF architecture's `context_length`, MLX `max_position_embeddings`
(preferentially under `text_config`), or consolidated/prequantized params'
`max_seq_len`. Missing or invalid native context metadata is an error.
`--max-prefill-chunk` independently bounds the positions processed per forward.
The runtime's `--max-context` controls the serving/cache budget and may be lower
than the exported native limit; exporting the full context does not require
allocating that full KV capacity at runtime.

## Build

On Apple Silicon, run `make muse-glimmer-batching-mlx` from the ExecuTorch
checkout. This builds and installs the MLX dependencies, then builds
`muse_glimmer_batching_runner` and `muse_glimmer_batching_worker` under
`cmake-out/examples/models/muse-glimmer-batching`. It does not run tests or model
exports. The existing `make muse-glimmer-mlx` target continues to build the
legacy runners.

Both `MUSE_GLIMMER_BUILD_BATCHING` and `MUSE_GLIMMER_BUILD_BATCHING_TESTS` default to
`OFF`. Either requires `EXECUTORCH_BUILD_MLX=ON`, an MLX-enabled ExecuTorch CMake
installation, and CUDA disabled. Preparation/materialization tests can be built
without the batching executor, but still require the MLX configuration.

Build and install ExecuTorch with the example's dependencies enabled. Both LLM
flags are necessary: the LLM option supplies batching/cache/serving, while the
runner option supplies the sampler and the retained legacy runners' dependency.
The following is an example configuration for an already prepared checkout:

```sh
cmake -S /path/to/executorch -B /path/to/build/executorch \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/install/executorch \
  -DEXECUTORCH_BUILD_MLX=ON \
  -DEXECUTORCH_BUILD_CUDA=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_TESTS=OFF
cmake --build /path/to/build/executorch --target install --parallel

cmake -S /path/to/executorch/examples/models/muse-glimmer \
  -B /path/to/build/mg-batching \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/install/executorch;/path/to/gflags" \
  -DEXECUTORCH_BUILD_MLX=ON \
  -DEXECUTORCH_BUILD_CUDA=OFF \
  -DMUSE_GLIMMER_BUILD_BATCHING=ON \
  -DMUSE_GLIMMER_BUILD_BATCHING_TESTS=OFF \
  -DMUSE_GLIMMER_STB_INCLUDE_DIR=/path/to/stb \
  -DMLX_METALLIB_PATH=/path/to/mlx.metallib
cmake --build /path/to/build/mg-batching \
  --target muse_glimmer_batching_runner muse_glimmer_batching_worker --parallel
```

Use a gflags CMake package prefix and an stb checkout containing both image and
legacy resize headers. The local stb override avoids a network fetch. Existing
registration-retention and MLX metallib-copy helpers are reused. Keep
`EXECUTORCH_BUILD_MLX=ON` in the example configuration as well as the dependency
build so the copy helper runs. Set `MLX_METALLIB_PATH` to the built metallib if
the installed package does not supply it.

To opt in to preparation/materialization tests, configure the example with
`-DMUSE_GLIMMER_BUILD_BATCHING_TESTS=ON` and build `muse_glimmer_batching_test`.
Building that target does not run it; test execution is a separate user step.

## Serve

```sh
python -m executorch.examples.models.muse_glimmer.serving.serve_batching \
  --worker-bin /path/to/build/mg-batching/muse_glimmer_batching_worker \
  --model-path /path/to/models/mg-batching/model.pte \
  --data-path /path/to/models/mg-batching/model.ptd \
  --pos-embed-path /path/to/models/mg-batching/pos_embed.bin \
  --tokenizer-path /path/to/models/mg/tokenizer.json \
  --hf-tokenizer /path/to/models/mg \
  --backend mlx \
  --max-context 16384
```

Omit `--data-path` when weights are embedded in the program. The launcher uses
one multiplexed worker process and the HF chat template through the existing MG
serving adapter, including Harmony formatting and optional ATEM tool parsing
(`--tool-parser atem`). It shuts the worker down through the async serving
lifecycle. It does not silently fall back to a legacy worker or DFlash.

Image requests use the OpenAI `image_url` content part with an inline base64
JPEG or PNG data URL. Remote URLs, file paths, `detail`, and more than one image
across the entire history are rejected. Resubmit the image inline when it is
part of the submitted history; no server-side image reference is supported.

Launcher limits mirror worker admission limits:

- `--max-inflight-requests` is a positive 32-bit integer (default 4), configuring
  admission capacity without a build-time concurrency ceiling. The worker advertises
  this capacity to the Python client. Session and decode-batch limits remain separate;
  choose all three according to available memory and desired concurrency.
- `--max-image-bytes` is 1 byte through 20 MiB (default 20 MiB), measured on the
  compressed image after base64 decoding, not on decoded pixels.
- `--max-request-bytes` is positive and at most 32 MiB (default 32 MiB), including
  input JSONL framing and the newline. It must be at least
  `4 * ceil(max_image_bytes / 3) + 1024`; limits can be lowered together.
- Output frames have a separate fixed 1 MiB limit, wired to the worker client.
- `--max-context` is greater than 1 and at most signed-int32 maximum; choose a
  value supported by the exported model. BOS/EOS IDs must fit unsigned uint64.

The base64/framing check is a minimum configuration check, not a guarantee that
an arbitrarily long text history fits. Whole-frame, decoded-pixel, vision-patch,
and model context limits still apply. Session and decode capacities default to
4; text prefix snapshots are opt-in via `--prefix-cache-entries` (default 0).

## Semantics

- Text-only prompts keep ordinary batching token-history/prefix reuse.
- Image prompts have immutable text/image layout and count image rows as decoder
  positions. They replace the whole session context and bypass prompt-prefix
  caching, rather than appending image bytes to a warm text prefix.
- The following text prompt cold-replays after image execution. KV cache state
  remains active within each generation.
- CPU prompt preparation runs after bounded admission on the control thread.
  Vision encoding, text embeddings, and decoder calls run on the engine thread.
  Prepared image embeddings are computed once per backing and may be sliced
  across physical forwards.
- Metadata/options/preparation validation occurs before replacement. A later
  engine/vision failure does not restore replaced state and follows the batching
  batch failure contract.

## Validation Status

Implementation validation is limited to static checks and C++ source
syntax/compile checks; these do not establish installed-package linking,
registration at runtime, metallib discovery, or model correctness. The current
handoff environment has no installed ExecuTorch CMake package, so a full batching
configure/build has not been validated there.

Launcher tests cover rejected limits, accepted boundaries, base64 framing, and
mocked worker lifecycle/input-output size wiring. They have not been run in this
handoff. No MG tests, exports, lowering, runtime, or batching binaries were run;
model-backed text/image generation and the commands above remain user-run
validation steps.
