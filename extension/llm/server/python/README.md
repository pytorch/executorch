# ExecuTorch LLM Server — Python

A thin OpenAI-compatible HTTP server over ExecuTorch's `LLMEngine`/`LLMSession`
serving API (with `TextLLMRunner` as the underlying adapter).

## Install

```bash
pip install -r requirements.txt
# transformers is optional but recommended for model-correct chat templates
pip install transformers
```

Requires an ExecuTorch build with the LLM runner pybindings
(`EXECUTORCH_BUILD_PYBIND=ON`) so `executorch.extension.llm.runner` imports.

### Model & runtime requirements

LLM `.pte` files exported via `export_llm` use ExecuTorch custom/quantized ops:
`use_sdpa_with_kv_cache` → `llama::custom_sdpa`, and quantized exports
(`embedding_quantize`, `8da4w`, ...) → `quantized_decomposed` ops. These are the
Python-runtime equivalent of the C++ build flags in the canonical
[Llama README](../../../../examples/models/llama/README.md)
(`-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON`,
`-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON`). **The server registers them
automatically** (imports `executorch.extension.llm.custom_ops.custom_ops` and
`executorch.kernels.quantized` before constructing the runner); without them the
runner fails with `Missing operator ... load_method('forward') failed`.

Tokenizer: pass the model's tokenizer — `tokenizer.json` (HF, e.g. Qwen3) or
`tokenizer.model` (Llama); the runner auto-detects. If you see an RE2 lookahead
warning it falls back to PCRE2 and still works (build with
`-DSUPPORT_REGEX_LOOKAHEAD=ON` for the native regex path).

## Run

```bash
python -m executorch.extension.llm.server.python.server \
    --model-path /path/to/model.pte \
    --tokenizer-path /path/to/tokenizer.bin \
    --hf-tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
    --model-id qwen2.5-coder \
    --enable-prefix-cache \
    --host 127.0.0.1 --port 8000
```

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
| `--num-runners N` | *requested* physical sessions (each = one KV cache, N × memory); the actual count is clamped by the engine's `serving_capacity()` — XNNPACK self-contained `.pte` is single-slot, so N>1 is clamped to 1 and extra requests queue |
| `--enable-prefix-cache` | opt-in turn-to-turn KV reuse (requires `--hf-tokenizer`; runs the LLMEngine/LLMSession path) |

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
# 1. Hermetic unit tests — fake engine, no model/GPU, fast (CI-friendly).
pip install pytest httpx
pytest tests/

# 2. Conformance — black-box, against a LIVE server (real model, or llama.cpp/mlx-lm).
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 pytest ../conformance/test_openai_contract.py
```

`tests/` swaps in a `FakeRunner` via `RunnerPool(runner_factory=...)`, so the real
server/protocol/streaming code is tested over HTTP without a `.pte`.

## Architecture

Control plane (this dir, Python): server, OpenAI protocol, chat templating,
session routing/streaming, and prefix-reuse *policy*. Data plane (C++): the
`LLMEngine`/`LLMSession` API owns token stepping and KV mutation (prefill/decode/
sampling) and releases the GIL. Python depends on `LLMEngine`/`LLMSession`, not on
`TextLLMRunner` token-step internals (`TextLLMRunner` is a legacy/direct runner
and a C++ implementation detail behind the session adapter). How many physical
sessions can exist without multiplying model memory is decided by
`serving_capacity()`, not by `--num-runners`. Tensor data never crosses into
Python element-wise.

| File | Role |
|------|------|
| `server.py` | FastAPI app, routes, CLI entrypoint |
| `protocol.py` | OpenAI request/response schemas |
| `chat_template.py` | messages (+tools) → prompt string |
| `runner_pool.py` | session pool + serving-capacity admission + affinity routing + async streaming bridge |
| `serving_chat.py` | `/v1/chat/completions` (streaming + non-streaming, stop, tools) |
| `prefix_cache.py` | turn-to-turn KV prefix-reuse policy over an `LLMSession` (opt-in) |
| `session_generate.py` | model-agnostic `LLMSession` → `generate()` adapter (no prefix reuse) |
| `tool_parsers/` | Hermes/Qwen `<tool_call>` parser only |

### Model adapters

The server is model-agnostic; the built-in path serves the text model
(`TextLLMEngine`). A new model implements the C++ `LLMEngine`/`LLMSession`
interfaces in its example, exposes them through the **generic pybind wrappers**
(`extension/llm/runner/llm_pybind_wrappers.h`) — **no per-model pybind class** —
and wires the generic server. The dependency points one way: a model example may
import the generic server; the generic server never imports an example. Backend
specifics (CUDA/AOTI; MLX/Metal is a future, non-validated extension point) stay
inside the model's engine.

A model that builds its own sessions and drives them through `RunnerPool`'s
`runner_factory` seam (e.g. with `SessionGenerateAdapter`) **must pass
`serving_capacity=` to `RunnerPool`** — the pool only auto-derives capacity from
an engine it owns, so a factory-backed pool that omits it could otherwise create
`--num-runners` physical sessions and silently duplicate weights. Capacity is
authoritative either way.

## Scope & caveats

Deliberately narrow (reliability-first): Hermes/Qwen tool calling only;
unsupported sampling params are rejected, not ignored. `--num-runners` is a
*request*, not a guarantee — the engine's `serving_capacity()` is authoritative,
and an XNNPACK self-contained `.pte` is conservative **single-slot** for v1
(packed weights may be per-method-instance, so extra physical sessions would
duplicate model memory): N>1 is clamped to 1 and concurrent requests queue on the
resident session. The engine serializes backend execution across sessions (op
kernels aren't assumed thread-safe — this is also what fixed the multi-runner
heap corruption). Prefix cache requires the LLMSession/engine path
(`--enable-prefix-cache` + `--hf-tokenizer`). Weight sharing across physical
sessions on a backend that supports it (e.g. CUDA/AOTI), adaptive thinking, and
multi-session subagents are future work.
