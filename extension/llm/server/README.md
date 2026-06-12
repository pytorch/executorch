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
structured API errors, and best-effort cancellation. One worker process with
serialized execution; it hosts many isolated sessions on one weight load (warm
append-only resume across turns). KV/prefix state lives inside the
worker/session, not the control plane. Unsupported params (including `top_p`,
`seed`, `n>1`, `reasoning_effort`, penalties, `logit_bias`, `response_format`,
`logprobs`, and `tool_choice="required"`) are rejected with a structured 400
rather than silently ignored. See `python/README.md` to run it and
`spec/README.md` for the exact contract.
