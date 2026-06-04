# ExecuTorch LLM Server

OpenAI-compatible serving for ExecuTorch LLMs, so any OpenAI-compatible agent
harness (pi, opencode, ...) can use ExecuTorch as a local backend.

```
extension/llm/server/
  spec/          # language-neutral OpenAI contract ExecuTorch targets
  conformance/   # one test suite every language server must pass
  python/        # Python server implementation (current)
  # cpp/         # future: no-Python single-binary server
```

Why this layout: the OpenAI contract is identical across languages, so the
**spec** and **conformance** suite are shared, and each language gets its own
implementation directory. The real cross-language reuse comes from the C++
`LLMEngine`/`LLMSession` primitives underneath, packaged as a process-isolated
**worker binary** (`text_llm_worker`) that any control plane drives over a small
JSONL protocol — the server is a thin protocol shell that spawns and talks to
that worker. See `python/README.md` to run it.

Status: experimental, reliability-first and deliberately narrow. Implemented:
`/health`, `/v1/models`, `/v1/chat/completions` (streaming + non-streaming),
Hugging Face chat templates (`--hf-tokenizer`), `temperature` / `max_tokens` /
`max_completion_tokens` / `stop`, Hermes tool calling by default
(`<tool_call>...</tool_call>` JSON, complete calls only; model-specific launchers
may select the Qwen XML format) with `tool_choice="none"`,
structured API errors, and best-effort cancellation. V1 serving is single-slot
(one worker, one session) with no prefix cache; KV prefix reuse, if it returns,
lives inside the worker/session, not the control plane. Unsupported params (including `top_p`,
`seed`, `n>1`, `reasoning_effort`, penalties, `logit_bias`, `response_format`,
`logprobs`, and `tool_choice="required"`) are rejected with a structured 400
rather than silently ignored. See `python/README.md` to run it and
`spec/README.md` for the exact contract.

## Use from pi (or any OpenAI-compatible harness)

Point pi at the server to use ExecuTorch as a local backend for tool-use
workflows. Launch the server:

```bash
python -m executorch.extension.llm.server.python.server \
  --model-path <model.pte> \
  --tokenizer-path <tokenizer.model-or-json> \
  --hf-tokenizer <hf-model-or-local-dir> \
  --model-id <model-id> \
  --host 127.0.0.1 \
  --port 8000
```

Useful optional flags (full reference in `python/README.md`):

- `--no-think` — default `enable_thinking=false` for templates that support it
  (e.g. Qwen3-style).
- `--max-context N` — reject over-long prompts cleanly; use the export-time
  context length.
- `--allow-chatml-fallback` — approximate ChatML when the model has no HF
  `chat_template`; experimentation only, not recommended for reliable tool use.

Point pi at the server via `~/.pi/agent/models.json`:

```json
{ "providers": { "executorch": {
    "baseUrl": "http://127.0.0.1:8000/v1", "api": "openai-completions",
    "apiKey": "x", "models": [ { "id": "<model-id>" } ] } } }
```

Other OpenAI-compatible clients use their own schema — generically: base URL
`http://127.0.0.1:8000/v1`, the model id you passed to `--model-id`, and a dummy
API key if one is required.

Supported contract for pi:

- Endpoint `POST /v1/chat/completions`; streaming supported.
- Tool calls: the model's Hermes-style `<tool_call>...</tool_call>` output is
  parsed and returned as OpenAI `tool_calls`. This generic server uses Hermes by
  default; a model-specific server may select the Qwen XML format.
- `tool_choice`: only `"auto"`, `"none"`, or unset.
- Rejected with a structured 400 (`unsupported_parameter`), not silently
  ignored: `tool_choice="required"` or specific-function forcing,
  `response_format` JSON/constrained output, `logprobs`, `top_p` other than
  `1.0`, and `seed`.

Reliability guidance:

- Use the model's real HF `chat_template` (`--hf-tokenizer`) for tool use, kept
  aligned with the exported tokenizer/model.
- If tool calls come back as plain text, confirm the model is emitting the
  configured tool-call format's markers (Hermes for the generic server) and that
  `tools` were included in the request.
- If a request fails with `unsupported_parameter`, remove or disable that
  OpenAI knob in your pi/client config.
