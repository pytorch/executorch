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

On the legacy path there is no global (cross-session) prefix cache: the control
plane holds no KV state and
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

## Multiplexed Native Worker (Opt-In)

This is a distinct text-only transport, not a capability of the legacy worker
loop. It requires a multiplex-aware client and exposes one `ServingRuntime`,
which owns one batching Runner. The adapter owns only JSONL framing and wire request
identities; tokenization, decoding, stop handling, and session history belong to
the runtime. Chat templates and OpenAI response presentation remain in Python.
The legacy cancellation and cross-session cache limitations above do not describe
this opt-in path. See the [batching-backed serving guide](batching.md) for launch
selection, creation-only caching, and native-backed Python integration checks.

Before accepting requests the worker emits one readiness record, with no request
ID:

```json
{"ready":true,"multiplexed":true,"max_named_sessions":4,"max_inflight_requests":64}
```

Despite its legacy name, `max_named_sessions` reports the shared capacity for
named and anonymous sessions; there is no additional reserved anonymous slot.

Every subsequent request has an explicit `op` and positive uint64 `request_id`.
Clients allocate IDs monotonically and guarantee lifetime uniqueness, but requests
may arrive out of order after reservation. The worker remembers only in-flight
operations, not a completed-ID set or watermark. Responses for different requests
may interleave and finish out of order; each admitted generation has zero or more
text records followed by exactly one terminal record. An admission error itself
is terminal and has no later runtime callback.

```json
{"op":"generate","request_id":8,"prompt":"Hello","max_new_tokens":32,"temperature":0,"top_p":1,"top_k":0,"seed":0,"stop":[]}
{"request_id":8,"token":" world"}
{"request_id":8,"done":true,"finish_reason":"length","cancelled":false,"prompt_tokens":1,"completion_tokens":32,"reused_prompt_tokens":0,"prefilled_prompt_tokens":1,"prefill_ms":2.0,"decode_ms":30.0,"total_ms":32.0,"prefill_tok_s":500.0,"decode_tok_s":1066.67,"session_reset_reason":""}
```

Generate accepts exactly one of `prompt` (string) and `prompt_segments` (array).
Segments are `{"text":"..."}` or `{"ids":[1,2]}`, with exactly one field;
IDs are nonnegative uint64 integers. Text segments
are encoded individually and IDs appended verbatim. Optional `session_id` names a
persistent session; it must be a nonempty string of at most 1024 UTF-8 bytes.
The full prompt is supplied each time, not a token delta. `max_new_tokens` is an
integer in `[-1, INT32_MAX]`; omitted or `-1` selects the runtime's automatic
budget. Sampling fields must be numeric, never booleans: temperature is finite
and nonnegative, top_p is in `(0, 1]`, top_k is a nonnegative int32, and seed is a
uint64 (zero preserves the existing unset/random sentinel). `stop` is an array
of strings. Unknown
fields and segment types, including images or image payload fields on otherwise
valid text segments, fail explicitly instead of silently dropping modalities.

Output JSON strings replace invalid/incomplete UTF-8 with U+FFFD, matching the
legacy worker's serialization policy; a generation ending mid-character does
not fail unrelated requests.

Optional `generated_token_ids` in a terminal is present only when the runtime can
safely replay exact output IDs. Statistics describe consumed prompt and generated
tokens, not necessarily visible text. Runtime failures have the terminal shape
`{"request_id":8,"error":"message","code":"internal"}`. Stable error codes
include `invalid_argument`, `not_ready`, `session_not_found`, `session_busy`,
`capacity_exhausted`, `internal`, `slow_consumer`, and `frame_too_large`.

Lifecycle commands are `open`, `close`, and `reset`, each with `session_id` and
its own request ID. Their replies are respectively `opened:true`, `closed:true`,
and `reset:true`, or a correlated error. They do not block the input reader.
An accepted close/reset fences its session: earlier generation text and terminals
precede its ACK, and runtime-owned generation callbacks and captures are cleaned
up before the ACK is enqueued. Later accepted same-session operations stay behind
that ACK, including commands submitted before it. Other sessions can progress.
Immediate validation/admission rejections and cancel replies are not covered by
this ordering. Close acknowledges logical release, not physical executor cleanup.
Reset acknowledges a cold replacement; a failed replacement does not restore the
old state.

Cancellation is in-band, not the legacy worker's separate FD pipe:

```json
{"op":"cancel","request_id":9,"target_request_id":8}
{"request_id":9,"cancelled":true}
{"request_id":8,"done":true,"cancelled":true,"finish_reason":"stop"}
```

The cancel acknowledgement is idempotent, including unknown or already completed
targets. It acknowledges the cancellation request, not generation completion.
The original generation finishes separately. Cancellation racing an already
selected completion does not rewrite that outcome.

### Bounds And Failure Policy

The default input/output record limit is 1 MiB including newline. The input reader
never accumulates an unbounded line. Invalid JSON, missing/invalid request IDs,
incomplete or oversized input records, and duplicate in-flight IDs fail the
transport. Duplicate IDs cannot receive an independent correlated error without
ambiguously terminating the original operation. Valid-ID field errors are isolated
to that request. EOF initiates runtime shutdown, drains generation and lifecycle
callbacks, and joins the Runner for physical cleanup before draining wire output.

Generation/lifecycle entries are bounded by `max_inflight_requests`, including
completed responses not yet written. A separate equally sized budget handles
cancel acknowledgements and capacity rejection responses; exhausting it fails the
transport. Every registered operation reserves one terminal/control record up to
the frame limit. Each generation additionally permits 64 queued token frames and
256 KiB of queued token bytes by default, plus at most one globally active writer
record. A full token queue requests cancellation only for its owning request,
drops its queued text, and waits for the runtime terminal before returning an
explicit `slow_consumer` terminal instead of a truncated success. This reports a
transport failure; it does not rewrite the runtime result or roll back committed
session history. An oversized terminal is replaced with `frame_too_large`, not
silently trimmed.

One writer emits queued JSONL records in FIFO order with bounded nonblocking POSIX
writes. Generation and lifecycle callbacks only enqueue; they never wait for
stdout. Runtime admission is released before those callbacks, so terminal output
does not wait for callback return or poll request handles. The serial input reader
retains its operation through handle assignment even if terminal output finishes
first, and replays any cancellation latched before assignment. Writing a terminal's
final newline and retiring its wire ID/capacity are atomic with respect to reader
admission. No lifecycle collector or per-command thread is needed.

If a record cannot finish writing within 10 seconds, stdout fails,
or the control reserve is exhausted, the entire transport fails and shuts down
the runtime. A permanently unread pipe cannot guarantee terminal delivery; clients
must settle all outstanding operations on EOF. Worker executables ignore SIGPIPE
and reserve stdout exclusively for protocol output.

### ModuleExecutor Construction And Deferred Smoke Test

The MLX example target `llm_worker` builds the shared ModuleExecutor bootstrap.
It requires a compatible packed, off-graph-cache text artifact and tokenizer. It
loads only the program before ModuleExecutor construction, reads activation dtype
and context length, resolves model/tokenizer EOS, binds the backend's default
batched cache via `cache::kind::kBatched`, derives scheduler width from
`preferred_batch_tokens()`, and constructs
one `ServingRuntime`. It does not preload `forward`, run token-step loops, or use
Glimmer's legacy session implementation. Prefix caching is disabled by default.
`--prefix_cache_entries=N` enables greedy creation-only reuse and adds `N + 1`
physical rows for retained snapshots and one transient capture; logical
`--max_sessions` remains unchanged.

With an existing MLX-enabled ExecuTorch installation and matching gflags package,
configure the MLX LLM examples and build `llm_worker`. Launch with `--pte`,
`--tokenizer`, `--max_sessions=4`, `--max_session_tokens=2048`, and
`--max_decode_sequences=2`; the last value must be smaller than the artifact's
packed forward width. Send two generate records without waiting for the first
terminal, observe both request IDs, cancel one, then verify the other still
completes and named-session continuation remains correct.

This exercises transport concurrency only. To establish physical GPU batching,
instrument a compatible ModuleExecutor forward and verify that one invocation
contains multiple session slices. Overlapping HTTP requests or sequential
single-session forwards are not proof. Real-model smoke testing is explicitly
deferred: no compatible local PTE/tokenizer has been confirmed, and this stack
must not download/export weights or claim that a real combined forward was
verified. A Glimmer `embed_text`/`forward_from_embeddings` last-logits artifact is
not assumed compatible with this packed construction path.
