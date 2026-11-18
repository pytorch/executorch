# Batching-Backed Text Serving

The multiplexed worker connects the existing OpenAI server to one shared
`batching::Runner`. Each persistent public key owns a move-only batching session;
anonymous requests get independent ephemeral sessions. The serving layer handles
text preparation, history reconciliation, rendering, and delivery. Scheduling and
speculative drafting/verification remain inside the runner and executor.

## Supported Boundary

- Prompts are full inputs, made from ordered text and exact token-ID segments.
  Text segments are encoded separately without implicit BOS/EOS. Chat templates
  must supply the model's required markers.
- Exact history extensions continue the existing session. With prefix caching
  enabled, replacements and retries can reuse matching shared snapshots. Clean
  replacements can also reuse a committed live prefix. Dirty live state is
  discarded; without a usable snapshot, the full supplied prompt is prefilled.
- Explicit open creates an empty session; opening an existing key preserves it.
  Explicit reset cancels outstanding generation and discards live state and
  history while retaining the session key. Failed reopen leaves the key
  unavailable. Shared snapshots survive open/reset and may be reused by the
  next generation when they match its supplied full prompt. Reset never adds
  old conversation content to that prompt and does not guarantee a cold prefill.
- Prefix caching is optional and disabled by default. It supports both greedy
  and sampled requests, including the first generation after explicit open or
  reset. `--prefix-cache-entries` reserves snapshot capacity in addition to
  logical `--max-sessions`. Set it to zero to disable shared prefix caching
  when a cold prefill is required after reset or opening a new key.
  Reuse requires a clone-capable executor and complete model state at the
  requested boundary; missing state falls back to prefill. Retained snapshots
  are shared by all clients, and `cached_tokens` reports matches, so enable
  prefix caching only when clients trust each other.
- Visible text is not model history. EOS and pending/internal tokens can remain
  in history without being visible. Omitted generated IDs mean replay is unsafe;
  an explicit empty list means a known empty completion.

- Different sessions run concurrently; requests within a session remain ordered.
  Callbacks must be nonblocking, and cancellation targets one request.
- Input support depends on the worker and export. The generic module worker
  accepts text; model-specific workers may also support images.

`--prefix-checkpoint-interval N` captures prompt snapshots at absolute multiples
of N and at the initial prompt end. It requires `--prefix-cache-entries > 0`.
Zero (the default) disables periodic capture.
`--prefix-checkpoint-prompt-end initial` (the default) captures the prompt end of
each request that creates a session, which includes every anonymous request.
`every` captures the prompt end of every request, such as tool-result turns and
retries. It requires `--prefix-cache-entries > 0` and cannot be combined with a
nonzero interval, whose serialized capture lane would drop prompt ends while
another checkpoint is pending. Capture is best effort. Without an interval, a prompt end already in the cache is
refreshed instead of cloned again, and a snapshot becomes reusable and frees its
capture slot at the source request's first output after the clone completes, or
when the request finishes. Retention uses global LRU; early checkpoints are not
protected, and with `every` each captured turn takes an entry.

A snapshot only helps a request whose prompt starts with the same tokens. The
Python adapters splice a named session's own generated tokens into its earlier
assistant turns, so `every` mainly helps that session after a turn that was
cancelled, cut short by a stop string, or failed. The retry or next turn then
reuses that turn's prompt if the interruption came after its first output token,
and otherwise the previous turn's prompt, instead of only the opening prompt that
`initial` keeps. Other sessions, anonymous requests, and sessions whose transcript
was dropped render those turns as text; with reasoning templates they usually
match that session's snapshots only up to the first assistant turn, which
`initial` already keeps. On MLX, snapshots share no KV storage: each keeps the
source's whole KV buffers, including growth slack and full sliding-window buffers,
and the source's next decode step copies them. Memory therefore grows with the
number and length of retained snapshots, so use fewer entries with `every` at long
contexts.

C++ applications can customize two independent policies in
[`ServingRuntimeConfig`](../../../extension/llm/serving/serving_runtime.h):

- `prefix_checkpoint_policy` chooses capture positions and optional application metadata,
  overriding `prefix_checkpoint_interval` and prompt-end capture. It cannot be
  combined with `PromptEndCapture::Every`. This C++ callback is not a launcher option.
- `prefix_cache_retention` chooses which snapshots to retain, including early anchors.

The [retention interface and default LRU policy](../../../extension/llm/batching/prefix_cache_retention.h)
also work with `PrefixCache` directly. Runner users can request exact boundaries
through `GenConfig::prefill_breakpoints` and `on_prefill`; see [runner.h](../../../extension/llm/batching/runner.h).

## Select the Worker Explicitly

Use `await spawn_multiplexed_worker(...)` with a worker advertising multiplexing
support. Create, use, and close the client on the same event loop.

For HTTP serving, use the dedicated launcher with compatible local artifacts.
This example uses Llama 3 chat formatting:

```bash
python -m executorch.examples.llm_server.python.serve \
  --worker-bin /path/to/llm_worker \
  --model-path /path/to/model.pte \
  --tokenizer-path /path/to/model-assets \
  --model-id llama1b \
  --max-context 1024 \
  --max-sessions 32 \
  --max-decode-sequences 16 \
  --assistant-header $'<|start_header_id|>assistant<|end_header_id|>\n\n'
```

Pass the HF tokenizer directory with its configuration, not an isolated JSON
file: native EOS resolution needs that configuration for warm continuation.
The launcher normalizes a supplied HF JSON file with sibling configuration to
its directory. The HF template source defaults to this directory; use
`--hf-tokenizer` to override it or when supplying another native tokenizer format.
An exact `--assistant-header` preserves the model's generation boundary; the
launcher probes it before spawning and warns if it does not match.

`--max-context` sets the native session-token and HTTP context limits together.
The launcher defaults are 16 logical sessions, 8 decode sequences per step, 64 in-flight
requests, and no prefix cache. Override the latter two with
`--max-inflight-requests` and `--prefix-cache-entries`. These limits are distinct:
Python concurrency follows the worker's advertised request capacity, not its
decode width or native delivery thread count. Request admission capacity is
independent of the single native delivery thread. Prefix-cache rows are additional
to logical session capacity.
The default HTTP address is `127.0.0.1:8000`; `--host` and `--port` override it.

The model must accept packed token/position inputs with per-token sequence IDs
and full or selected logits. A leading tensor dimension of one does not imply
single-session execution: sessions can be packed along the token dimension.
Last-token-only logits and single-session embedding-forward exports are not
substitutes for this contract. The native bootstrap loads the program, binds
runtime cache state, and only then loads the forward method.

## CPU Integration Checks

With a matching POSIX ExecuTorch installation built with
`EXECUTORCH_BUILD_EXTENSION_LLM=ON`, `EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON`,
and `EXECUTORCH_OPTIMIZE_SIZE=OFF`, build the test-only worker and run the opt-in
Python checks:

```bash
cmake -S /path/to/executorch/examples/llm_server/cpp \
  -B /tmp/et-serving-worker-tests \
  -DCMAKE_PREFIX_PATH=/path/to/executorch-install \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/et-serving-worker-tests \
  --target test_batching_worker test_multiplexed_worker --parallel 4
ctest --test-dir /tmp/et-serving-worker-tests \
  -R '^multiplexed_worker$' --output-on-failure --timeout 30
EXECUTORCH_BATCHING_TEST_WORKER=/tmp/et-serving-worker-tests/test_batching_worker \
  python -m pytest \
  /path/to/executorch/examples/llm_server/python/tests/test_batching_end_to_end.py -q
```

The tests need the server's HTTP/test dependencies and an importable checkout,
but no model weights or GPU. They cover serving, batching, cancellation, and
prefix reuse with a fake executor.

## Real-Model Checks

The launcher uses existing local artifacts; it does not export or download model
weights. One compatible Llama export recipe is:

```bash
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
  --model-id unsloth/Llama-3.2-1B-Instruct \
  --output /path/to/llama1b_offgraph.pte \
  --dtype bf16 \
  --use-offgraph-cache \
  --max-ctx-len 1024 \
  --prefill-chunk-size 512 \
  --logits-to-keep full
```

Build `llm_worker` from the MLX example project against a matching MLX-enabled
ExecuTorch installation. Use the export's tokenizer and chat template. Start
with caching disabled and greedy sampling, issue at least two overlapping
requests, and instrument the executor/backend to confirm multiple session IDs
occur in one actual model forward. Correct concurrent responses alone are not
sufficient evidence. Then check targeted cancellation and persistent continuation.

Prefix-cache testing is a separate opt-in step. MLX cold/warm logits and tokens
can differ because reuse changes prefill shapes; greedy sampling does not
promise numerical or token parity. Check correctness and reuse accounting
without treating identical output as the sole acceptance criterion.
