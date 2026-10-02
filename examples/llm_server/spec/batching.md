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
- Exact strict history extensions reuse the existing session. Equal, mismatched,
  or dirty histories cold-replay. Reset destroys the old session and reopens
  under the same reserved key; failed reopen leaves that key unavailable.
- Prefix caching is optional and disabled by default. It is only for greedy
  implicit new-session initialization, never ordinary continuation, explicit
  open, reset, or replay. Snapshot capacity is additional to logical serving
  capacity. The worker's `--prefix_cache_entries` option reserves those snapshots
  plus one transient capture row without reducing logical `--max_sessions`.
- Visible text is not model history. EOS and pending/internal tokens can remain
  in history without being visible. Omitted generated IDs mean replay is unsafe;
  an explicit empty list means a known empty completion.
- Different sessions can run concurrently. Same-session transcript preparation,
  generation, native wire completion, and transcript commit remain ordered.
  Cancellation targets one request. Native output and completion-preparation
  callbacks run in request order on one shared delivery thread, never the engine
  or control thread. They must do short, bounded, nonblocking work: no blocking
  I/O, synchronous runtime shutdown/destruction, or waits for runtime work.
  A blocked native callback delays other native delivery. The native transport
  enqueues bounded output and writes pipes separately. Python callbacks run on
  consuming threads, never the stdout reader; slow Python callbacks remain
  isolated from other requests and consume bounded capacity.
- Image/audio preparation is not implemented by this worker. The existing
  muse-glimmer image-serving path remains on its legacy worker. This change does
  not make existing glimmer embedding-forward exports compatible with the packed
  `ModuleExecutor` or implement solo/DFlash executors for MLX or CUDA.

## Select the Worker Explicitly

`spawn_worker` selects multiplexing only from a positive `multiplexed: true`
readiness capability. Ordinary callers retain legacy fallback. A launch that
requires batching must pass `require_multiplexing=True` so an older worker is
rejected rather than silently serialized.

The following is a launch recipe, not a completed real-model smoke test. Replace
all `/path/to` paths with absolute paths to matching local artifacts:

```python
import uvicorn

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.worker_client import spawn_worker

worker = spawn_worker(
    [
        "/path/to/llm_worker",
        "--pte=/path/to/model.pte",
        "--tokenizer=/path/to/model-assets/tokenizer.json",
        "--max_sessions=4",
        "--max_session_tokens=1024",
        "--max_decode_sequences=2",
    ],
    require_multiplexing=True,
)
runtime = SessionRuntime(worker)
template = ChatTemplate(hf_tokenizer_path="/path/to/model-assets")
serving = ServingChat(runtime, template, "local-model", max_context=1024)
try:
    uvicorn.run(build_app(serving, "local-model"), host="127.0.0.1", port=8000)
finally:
    runtime.close_worker()
```

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

The Python environment needs the example server's HTTP/test dependencies and an
importable checkout. The async tests use AnyIO's asyncio backend. No model weights
or GPU are needed. The test worker uses the real transport, serving runtime, and
runner with a fake executor and byte tokenizer. Its test-only scheduler gate
waits for two queued session IDs without blocking the engine, and its trace
records distinct session IDs in each `execute` call.

These checks prove concurrent admission and a multi-session executor batch,
including requests through the OpenAI endpoint. They also exercise cancellation,
slow Python consumer isolation, exact-ID continuation, reset/close, out-of-order reserved
request IDs, creation-only prefix caching through the Python runtime, and
unknown-versus-empty replay metadata. They do not prove a combined real-model
forward.

## Deferred Real-Model Smoke Test

No weights were downloaded or exported for this work. Real-model execution is
explicitly deferred. One compatible export recipe to evaluate later is:

```bash
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
  --model-id unsloth/gemma-3-1b-it \
  --output /path/to/gemma3_offgraph.pte \
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
