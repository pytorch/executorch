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
`stream`, `temperature`, `top_p`, `top_k`, `seed`, `max_tokens` /
`max_completion_tokens`, `stop`, `tools`, `tool_choice` (only `"none"` to disable
tools, or `"auto"`/unset for default parsing), `stream_options.include_usage`, and
`chat_template_kwargs` (e.g. `enable_thinking`). `top_p` defaults to `1.0`, `top_k`
defaults to `0` (disabled), and an explicit `seed` must be positive; omitted seed
uses the worker's unset/random value.
`model` must match the id returned by `/v1/models`; unknown ids return
`404 model_not_found`.

`chat_template_kwargs.return_reasoning` is an ExecuTorch response-visibility
control and defaults to `true`. For models with a reasoning extractor, set it
to `false` to omit reasoning from the response; the model still computes reasoning.
Unlike disabling reasoning separation in SGLang or llama.cpp, this suppresses
extracted reasoning instead of leaving it in `content`.

The flag must be a JSON boolean for every model, including models without a
reasoning extractor. Non-boolean values return `400 invalid_request_error`
(`code: "invalid_value"`) before session admission or generation. This tightens
earlier behavior, which accepted non-boolean values as an opt-out.

**Rejected** with `400 invalid_request_error` (`code: "unsupported_parameter"`)
rather than silently ignored — a client relying on them would otherwise get
wrong behavior: `n` (> 1), `reasoning_effort`,
`frequency_penalty`/`presence_penalty` (nonzero), `logit_bias`, `tool_choice` =
`"required"` or a specific-function choice
(forcing/restricting a call needs constrained decoding, not implemented),
`response_format` other than `{"type": "text"}` (no constrained JSON),
`logprobs`/`top_logprobs` (not returned), and `parallel_tool_calls: false`
(single-call can't be guaranteed without constraining). Unknown fields that
don't affect the output (e.g. `user`, `store`, `metadata`) are accepted and
ignored.

Non-streaming response: `chat.completion` with one `choice`
(`message.role = "assistant"`, string `content` or `tool_calls`, `finish_reason`
∈ `stop` | `length` | `tool_calls`) and a `usage` block.
When reasoning is returned, `message.reasoning_content` contains the extracted
reasoning text separately from visible `content` and `tool_calls`.
`usage.prompt_tokens_details.cached_tokens` reports prompt tokens served from
the session's resident state instead of prefetched this request (0 when the
turn fully prefilled); the streaming usage chunk carries the same field.

Streaming response: `text/event-stream` of `chat.completion.chunk` objects —
first chunk carries `delta.role = "assistant"`, subsequent chunks carry
`delta.content` (or buffered `delta.tool_calls`), a final chunk carries
`finish_reason`, optionally a usage-only chunk (with
`stream_options.include_usage`), terminated by `data: [DONE]`.
For models with a reasoning extractor, output is buffered and returned reasoning
is emitted as `delta.reasoning_content` before visible content or tool calls.

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
the worker runs the in-flight request to completion — there is no mid-generation
interrupt protocol. Because execution is serialized on one worker, an abandoned
generation also **head-of-line blocks** every other session until it finishes;
real interruption (a control pipe / between-step stdin poll / request-id cancel
op) is future work.

### Prefix / KV reuse

No global (cross-session) prefix cache: the control plane holds no KV state and
does no prefix-reuse routing, so a system prompt shared by two different sessions
is prefilled independently for each. Per-session append-only warm resume *is*
implemented worker-side for engines that support it — a named session whose next
request is an exact-token extension of its resident context prefills only the new
suffix. All KV/resident state lives inside the worker/session, never the control
plane.

For named sessions, an unchanged assistant reply can reuse its original generated
token IDs. Clients may omit `reasoning_content` or send `null` when echoing a reply;
both allow replay of the original tokens, including reasoning. A supplied string
must exactly match the value returned to that client. String edits, including
whitespace or line-ending changes and `""`, invalidate that turn's stored IDs and
later records. Invalidated turns render normally and do not regain their old IDs
if the client restores the old history. Newly generated turns can be replayed at
their original assistant-turn indices.

If an assistant boundary cannot be verified, the server uses the rendered text.
The generic launcher accepts `--assistant-header` and warns at startup if it is
absent from the template's generation prompt. The worker checks the resulting
token sequence before reusing KV state in either case; text fallback may still
reuse KV when the tokens match.
