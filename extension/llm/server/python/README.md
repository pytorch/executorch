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
| `--num-runners N` | V1 supports **1 only** (single-slot: one worker serves one session; concurrent requests queue) |
| `--worker-bin PATH` | path to the `text_llm_worker` binary (default: `cmake-out/extension/llm/server/cpp/text_llm_worker`) |

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
      "apiKey": "x", "models": [ { "id": "qwen2.5-coder" } ] } } }
  ```

## Validate

Two layers, both contract-focused (assert on the wire, not internals):

```bash
# 1. Hermetic unit tests — fake worker, no model/GPU/subprocess, fast (CI-friendly).
pip install pytest httpx
pytest tests/

# 2. Conformance — black-box, against a LIVE server (real model, or llama.cpp/mlx-lm).
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 pytest ../conformance/test_openai_contract.py
```

`tests/` builds a `RunnerPool` over a single `FakeRunner` worker handle, so the
real server/protocol/streaming code is tested over HTTP without a `.pte`. The
worker JSONL protocol is covered separately by `tests/test_worker_client.py`.

## Architecture

Control plane (this dir, Python): server, OpenAI protocol, chat templating,
streaming bridge, tool parsing — no CUDA, no model, no pybind. Data plane (C++):
a worker process (`text_llm_worker`) owns one model session and does all token
stepping and KV mutation; it speaks one JSON object per line on stdin/stdout.

JSONL protocol (stdout carries protocol JSON only; logs go to stderr):

```
worker -> stdout, once at startup:  {"ready": true}
client -> stdin,  per request:      {"prompt", "max_new_tokens", "temperature"}
worker -> stdout, per request:      {"token": str} *        (streamed)
                                    {"done": true, "prompt_tokens", "completion_tokens"}
                                or  {"error": str}
```

Process isolation is the reliable shape for CUDA/AOTI models: executing the model
inside a live asyncio server process can segfault (validated with Qwen3.5-MoE);
the worker is a plain process with no asyncio loop, and the control plane only
does blocking pipe I/O on its executor thread.

| File | Role |
|------|------|
| `server.py` | FastAPI app, routes, CLI entrypoint, worker spawn |
| `protocol.py` | OpenAI request/response schemas |
| `chat_template.py` | messages (+tools) → prompt string |
| `worker_client.py` | spawn a worker process + drive it over JSONL |
| `runner_pool.py` | worker pool (one in-flight request per worker) + async streaming bridge |
| `serving_chat.py` | `/v1/chat/completions` (streaming + non-streaming, stop, tools) |
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
unsupported sampling params are rejected, not ignored. V1 is **single-slot**: one
worker hosts one session, so `--num-runners` accepts 1 and concurrent requests
queue. Serving capacity is worker capacity, chosen by the launcher (each worker
is its own process with its own weights, so N workers cost N × the weight memory)
— an operator decision, not something the pool infers.

Cancellation is best-effort: a worker request runs to completion and is not
interruptible mid-generation in V1, so `runner.stop()` means "the control plane
stops consuming and the worker finishes the current request" rather than a hard
cancel. There is **no prefix cache in V1 serving**; if KV prefix reuse returns it
will live inside the worker/session, not in the Python control plane. Multiple
workers, weight sharing across sessions on a backend that supports it, adaptive
thinking, and multi-session subagents are future work.
