# ExecuTorch LLM Server — Python

A thin OpenAI-compatible HTTP server for ExecuTorch LLMs. The Python process is
the **control plane** only — HTTP, OpenAI protocol, chat templating, tool
parsing, request validation. Model execution runs in a separate **C++ worker
process** (`text_llm_worker`) that the server drives over a small JSONL protocol.
The control plane never loads a model, links a backend, or imports a runtime
pybind.

## Install

```bash
pip install -r requirements.txt
# transformers is optional but recommended for model-correct chat templates
pip install transformers
```

The server itself is pure Python (fastapi, pydantic, httpx). The model runs in
the C++ worker, which you build standalone (like the example runners) from
`../cpp`:

```bash
cmake -S ../cpp -B <cmake-out>/extension/llm/server/cpp \
      -DCMAKE_PREFIX_PATH=<cmake-out> -DEXECUTORCH_BUILD_XNNPACK=ON
cmake --build <cmake-out>/extension/llm/server/cpp --target text_llm_worker
# -> <cmake-out>/extension/llm/server/cpp/text_llm_worker
```

### Model & runtime requirements

LLM `.pte` files exported via `export_llm` use ExecuTorch custom/quantized ops:
`use_sdpa_with_kv_cache` → `llama::custom_sdpa`, and quantized exports
(`embedding_quantize`, `8da4w`, ...) → `quantized_decomposed` ops. The worker
binary links the kernel libraries that provide them (the C++ equivalents of
`-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON` /
`-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON`); see the canonical
[Llama README](../../../../examples/models/llama/README.md). Without them the
worker fails to load the method.

Tokenizer: pass the model's tokenizer — `tokenizer.json` (HF, e.g. Qwen3) or
`tokenizer.model` (Llama); the worker auto-detects. If you see an RE2 lookahead
warning it falls back to PCRE2 and still works (build with
`-DSUPPORT_REGEX_LOOKAHEAD=ON` for the native regex path).

## Run

```bash
python -m executorch.extension.llm.server.python.server \
    --model-path /path/to/model.pte \
    --tokenizer-path /path/to/tokenizer.bin \
    --hf-tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
    --model-id qwen2.5-coder \
    --host 127.0.0.1 --port 8000
```

The server spawns the worker (it blocks until the worker has loaded the model and
reported ready, so a slow load surfaces at startup, not on the first request).

`--hf-tokenizer` is **required** (it applies the model's real `chat_template`)
unless you pass `--allow-chatml-fallback` to opt into approximate generic ChatML
— which is wrong for many instruct/tool models and can't reproduce controls like
`enable_thinking`.

Key flags:

| Flag | Effect |
|------|--------|
| `--hf-tokenizer` | model's HF chat template (required unless fallback) |
| `--allow-chatml-fallback` | opt into approximate ChatML when no HF tokenizer |
| `--no-think` | default `enable_thinking=False` (e.g. Qwen3) |
| `--max-context N` | reject over-long prompts with 400 instead of failing mid-gen |
| `--num-runners N` | Worker processes — **1 only** (one worker hosts many isolated sessions on one weight load; more would duplicate weights) |
| `--worker-bin PATH` | path to the `text_llm_worker` binary (default: `cmake-out/extension/llm/server/cpp/text_llm_worker`) |

## Smoke test

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"hello"}]}'
```

## Sessions

When the worker reports named-session capacity (a worker whose engine supports it,
launched with `--max-sessions N >= 2`; the generic `text_llm_worker` reports
none), a request can target a persistent per-conversation session:

- body `session_id`, or headers `X-ExecuTorch-Session-ID` / `session_id` /
  `x-session-affinity` (body wins) — a stable id reuses that session's KV across
  turns (warm resume).
- `POST /v1/sessions/{id}/reset` — clear its context, keep the slot.
- `DELETE /v1/sessions/{id}` — free its context and slot.

## Use from an agent harness

- **opencode** (`opencode.json`):
  ```json
  { "provider": { "executorch": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": { "qwen2.5-coder": { "name": "Qwen2.5-Coder (ExecuTorch)" } } } } }
  ```
- **pi** (`~/.pi/agent/models.json`):
  ```json
  { "providers": { "executorch": {
      "baseUrl": "http://127.0.0.1:8000/v1", "api": "openai-completions",
      "apiKey": "x", "models": [ { "id": "qwen2.5-coder",
        "compat": { "sendSessionAffinityHeaders": true } } ] } } }
  ```
  `compat.sendSessionAffinityHeaders` makes pi route each conversation to its own
  session (per-conversation isolation + warm resume); without it every request
  uses the anonymous scratch session.

## Validate

Two layers, both contract-focused (assert on the wire, not internals):

```bash
# 1. Hermetic unit tests — fake worker, no model/GPU/subprocess, fast (CI-friendly).
pip install pytest httpx
pytest tests/

# 2. Conformance — black-box, against a LIVE server (real model, or llama.cpp/mlx-lm).
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 pytest ../conformance/test_openai_contract.py
```

`tests/` builds a `SessionRuntime` over a single `FakeRunner` worker, so the
real server/protocol/streaming code is tested over HTTP without a `.pte`. The
worker JSONL protocol is covered separately by `tests/test_worker_client.py`.

## Architecture

Control plane (this dir, Python): an OpenAI adapter (`serving_chat`) over a
stateful `SessionRuntime` over one `WorkerClient` — server, protocol, chat
templating, streaming bridge, tool parsing — no CUDA, no model, no pybind. Data
plane (C++): a worker process (`text_llm_worker`) that owns all model state
(many isolated sessions on one weight load, warm-resume prefix logic) and does
all token stepping and KV mutation; it speaks one JSON object per line on
stdin/stdout.

The JSONL protocol — `generate` / `open` / `close` / `reset` ops, the `prompt` /
`prompt_segments` prompt forms, warm-resume stats, and `generated_token_ids` — is
defined in `cpp/worker_loop.h` (the worker side, the canonical reference) and
driven by `worker_client.py` (the Python transport); stdout carries protocol JSON
only, logs go to stderr.

Process isolation is the reliable shape for CUDA/AOTI models: executing the model
inside a live asyncio server process can segfault (validated with Qwen3.5-MoE);
the worker is a plain process with no asyncio loop, and the control plane only
does blocking pipe I/O on its executor thread.

| File | Role |
|------|------|
| `server.py` | FastAPI app, routes, CLI entrypoint, worker spawn |
| `protocol.py` | OpenAI request/response schemas |
| `chat_template.py` | messages (+tools) → prompt string |
| `worker_client.py` | spawn a worker process + drive it over JSONL (raw transport) |
| `session_runtime.py` | stateful runtime over one worker: open/generate/reset/close + streaming bridge |
| `openai_transcript.py` | OpenAI token-ID warm-resume state (fingerprints + sentinel splicing) |
| `serving_chat.py` | `/v1/chat/completions` OpenAI adapter (streaming + non-streaming, stop, tools) |
| `tool_parsers/` | Hermes/Qwen `<tool_call>` parser only |
| `cpp/text_llm_worker.cpp` | the generic C++ worker binary |

### Model workers

The generic `text_llm_worker` serves the text model (`TextLLMEngine`). A new
model ships its own worker binary under its example (e.g.
`examples/models/qwen3_5_moe/qwen35_moe_worker.cpp` constructs `Qwen35MoEEngine`)
that speaks the same JSONL protocol, plus a launcher that points the same control
plane at that binary via `--worker-bin`. The dependency points one way: a model
example may reuse the generic control plane; the generic control plane never
imports an example. Backend specifics (CUDA/AOTI, Metal) stay inside the worker.

## Scope & caveats

Deliberately narrow (reliability-first): Hermes/Qwen tool calling only;
unsupported sampling params are rejected, not ignored. **One worker process,
serialized execution** (one in-flight request; concurrent requests queue).
Session capacity is determined by the worker/engine — a single worker hosts many
isolated sessions on one weight load — so `--num-runners` accepts 1; extra worker
processes would each carry their own copy of the weights.

The **generic `text_llm_worker` is scratch-only**: `TextLLMEngine::serving_capacity()`
is a conservative 1, so `max_named = max(0, capacity-1) = 0` — the default
`server.py` serves only the anonymous scratch session (no named `session_id`s, no
warm resume). The named-session / warm-resume / token-ID machinery is exercised
by a model-specific worker whose engine reports capacity > 1 (the Qwen3.5-MoE CUDA
worker). This is intentional; the generic worker stays minimal until a backend is
proven to host multiple physical sessions without duplicating weights.

**Cancellation is best-effort, and it head-of-line blocks.** `WorkerClient.stop()`
is a no-op, and `SessionRuntime.generate_stream()` holds the single worker lock until
the worker naturally finishes. On client disconnect/cancellation the server calls
`stop()` then awaits the in-flight worker request, so the abandoned generation runs to
completion **and blocks every other session on that worker until it does** — a long or
runaway generation stalls all concurrent requests (including a subagent fan-out).
A disconnected client does **not** interrupt the C++ worker mid-generation. Real
interruption needs a future protocol change — e.g. a control pipe, non-blocking stdin
polling between decode steps, or request ids plus an out-of-band cancel op.

**Warm resume needs true turn terminators surfaced as EOS/terminal token ids, not just
string stops.** The worker treats every *string* stop the same — it trims the output,
marks the session dirty, and omits `generated_token_ids` — which is correct for
user/request stops and broad content-cleanup stops (they change visible text, so the
turn is non-resumable). A clean model turn terminator is only resumable if the engine
surfaces it as a terminal/EOS **token id** (the Qwen engine adds `<|im_end|>` to
`eos_ids`, so it ends the turn before string-stop matching and stays resumable). A
backend whose terminator is only a string stop would mark every turn dirty and never
warm-resume; distinguishing resumable terminators from trim-stops in the protocol is
future work.

There is **no global (cross-session) prefix cache**; per-session append-only warm
resume is worker-side (for engines that support it), and all KV/resident state
lives inside the worker/session, never the Python control plane. Multiple workers,
weight sharing across sessions on a backend that supports it, adaptive thinking,
and multi-session subagents are future work.
