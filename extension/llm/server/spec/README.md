# ExecuTorch LLM Server — Contract Spec

The language-neutral contract every ExecuTorch LLM server (Python today, C++
later) implements. The conformance suite in `../conformance` validates an
implementation against this spec by hitting a live server, so it is independent
of language and engine.

## Supported endpoints

| Endpoint | Status |
|----------|--------|
| `GET /v1/models` | implemented |
| `POST /v1/chat/completions` (stream + non-stream) | implemented |
| `GET /health` | implemented |
| `POST /v1/completions` | planned |

## `POST /v1/chat/completions`

OpenAI Chat Completions subset. **Honored** request fields: `model`, `messages`,
`stream`, `temperature`, `max_tokens` / `max_completion_tokens`, `stop`, `tools`,
`tool_choice` (only `"none"` to disable tools, or `"auto"`/unset for default
parsing), `stream_options.include_usage`, and `chat_template_kwargs` (e.g.
`enable_thinking`).

**Rejected** with `400 invalid_request_error` (`code: "unsupported_parameter"`)
rather than silently ignored — a client relying on them would otherwise get
wrong behavior: `top_p` (anything other than `1.0`), `seed`, `n` (> 1),
`reasoning_effort`, `frequency_penalty`/`presence_penalty` (nonzero), `top_k`,
`logit_bias`, `tool_choice` = `"required"` or a specific-function choice
(forcing/restricting a call needs constrained decoding, which v1 lacks),
`response_format` other than `{"type": "text"}` (no constrained JSON),
`logprobs`/`top_logprobs` (not returned), and `parallel_tool_calls: false`
(single-call can't be guaranteed without constraining). Unknown fields that
don't affect the output (e.g. `user`, `store`, `metadata`) are accepted and
ignored.

Non-streaming response: `chat.completion` with one `choice`
(`message.role = "assistant"`, string `content` or `tool_calls`, `finish_reason`
∈ `stop` | `length` | `tool_calls`) and a `usage` block.

Streaming response: `text/event-stream` of `chat.completion.chunk` objects —
first chunk carries `delta.role = "assistant"`, subsequent chunks carry
`delta.content` (or buffered `delta.tool_calls`), a final chunk carries
`finish_reason`, optionally a usage-only chunk (with
`stream_options.include_usage`), terminated by `data: [DONE]`.

### Tool calling

Two output formats are accepted: Hermes-style JSON
(`<tool_call>{"name":...,"arguments":{...}}</tool_call>`, used by Qwen2.5/Qwen3)
and Qwen XML-style (`<function=NAME><parameter=K>V</parameter></function>`,
typically wrapped in `<tool_call>`, used by Qwen3.5-MoE / Qwen3-Coder). The
server buffers the model's full output and emits **complete** OpenAI
`tool_calls` (no partial-argument fragments). Calls to tools absent from the
request, and malformed tool calls, degrade to visible text — never a crash or
silent drop. `tool_choice="none"` disables tool parsing.

### Errors & cancellation

Errors return `{"error": {"message", "type", "code"}}` with an appropriate
status (e.g. `400 context_length_exceeded` when `--max-context` is set and the
prompt exceeds it). A mid-stream failure emits an `error` SSE event then
`[DONE]` rather than dropping the socket. Cancellation is best-effort: on a
client disconnect the control plane stops consuming the stream (`stop()`), but
the worker runs the in-flight request to completion — V1 has no mid-generation
interrupt protocol.

### Prefix cache

Not in V1 serving. The control plane holds no KV state and does no prefix-reuse
routing; each request is an independent prompt to the worker. If turn-to-turn KV
prefix reuse returns, it will live inside the worker/session (where the KV cache
is), not in the control plane.
