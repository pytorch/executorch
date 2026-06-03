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
`LLMEngine`/`LLMSession` primitives underneath (with `TextLLMRunner` as the
current adapter) — each server is a thin protocol shell over that engine. See
`python/README.md` to run it.

Status: experimental, reliability-first and deliberately narrow. Implemented:
`/health`, `/v1/models`, `/v1/chat/completions` (streaming + non-streaming),
Hugging Face chat templates (`--hf-tokenizer`), `temperature` / `max_tokens` /
`max_completion_tokens` / `stop`, Hermes/Qwen tool calling
(`<tool_call>...</tool_call>`, complete calls only) with `tool_choice="none"`,
structured API errors, cancellation, and an opt-in conservative per-runner KV
prefix cache (`--enable-prefix-cache`). Unsupported params (including `top_p`,
`seed`, `n>1`, `reasoning_effort`, penalties, `logit_bias`, `response_format`,
`logprobs`, and `tool_choice="required"`) are rejected with a structured 400
rather than silently ignored. See `python/README.md` to run it and
`spec/README.md` for the exact contract.
