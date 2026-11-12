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

Each `/v1/models` entry includes `context_length` when the server has a configured
context limit. Clients must use this value or an explicit override for compaction
budgeting; advertising it does not make every client consume it automatically.

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

`messages` must be nonempty and roles must be `system`, `developer`, `user`,
`assistant`, or `tool`. Tool declarations require unique nonempty function names
and object parameter schemas (omitted/null means an empty schema). Historical tool calls require an ID, name, and JSON
argument object (omitted/null/empty arguments mean `{}`). Call IDs must be unique
within an assistant message, but may be reused across turns. Tool results require
a `tool_call_id`; they need not have a matching call in the supplied history.
Compacted histories and repeated tool results are preserved for template rendering.
Malformed structures and boolean/string values for supported numeric generation controls
fail schema validation. Integer controls also reject floats, including integral floats
such as `1024.0`. These failures return HTTP 422 and an `error` object before inference. Unsupported
penalty/logprob controls retain their existing Pydantic coercion before capability checks.

`ignore_eos` is an optional JSON boolean (default `false`) supported by the
multiplexed batching worker. It disables model EOS and automatic chat-template
stops, allowing generation through terminator tokens up to the requested budget
or context limit. Explicit request `stop` strings and cancellation still apply.
Response parsing ends at the first decoded turn terminator from the worker or
chat template. Plain chat also treats content-special tokens as visible boundaries;
under `ignore_eos`, these boundaries hide text without stopping generation.
If a request stop cuts through a special-token spelling, an ambiguous trailing
fragment is hidden too, even if it came from literal text.
The worker advertises the decoded strings of all its default EOS
IDs; HTTP `ignore_eos` is rejected if that metadata is absent or any EOS marker
cannot decode to a nonempty string. Use matching model/tokenizer assets. Tokens
past a recognized terminator contribute to usage but are excluded from parsing. Explicit
stops still apply to hidden text, and exhausting the budget reports `length` even
if the visible answer ended earlier. Exact generated-token splicing is disabled
for these turns. A later prompt rebuilt from the visible answer can differ from
the resident history containing hidden tokens, so direct continuation may reset
and re-prefill. Enabled prefix caching can recover a matching retained prefix.
AgentX synthesizes later prompts independently; enable prefix caching when
benchmarking reuse. Explicit `max_tokens` is recommended;
without it generation uses the remaining context budget. The worker advertises
support and `default_stop_strings` in its readiness frame (`null` when unavailable).
Older and legacy workers reject
`ignore_eos: true` with `400 unsupported_parameter`. Non-boolean values, including
null, fail schema validation with HTTP 422.

`chat_template_kwargs.return_reasoning` is an ExecuTorch response-visibility
control and defaults to `true`. For models with a reasoning extractor, set it
to `false` to omit reasoning from the response; the model still computes reasoning.
Unlike disabling reasoning separation in SGLang or llama.cpp, this suppresses
extracted reasoning instead of leaving it in `content`.

`include_reasoning` is an optional JSON boolean alias for this visibility
control, honored in both API dialects and both streaming modes. Omitted/null
uses the existing default. Conflicting explicit visibility settings return
`400 invalid_value`, including conflicts with a model adapter's
`reasoning.exclude`; non-boolean `include_reasoning` fails schema validation.

The flag must be a JSON boolean for every model, including models without a
reasoning extractor. Non-boolean values return `400 invalid_request_error`
(`code: "invalid_value"`) before session admission or generation. This tightens
earlier behavior, which accepted non-boolean values as an opt-out.

**Rejected** with `400 invalid_request_error` (`code: "unsupported_parameter"`)
rather than silently ignored — a client relying on them would otherwise get
wrong behavior: `n` (> 1), `reasoning_effort` and OpenRouter `reasoning`
(except `{}` and `{"exclude": false}`, or options supported by a model-specific adapter),
`frequency_penalty`/`presence_penalty` (nonzero), `logit_bias`, `tool_choice` =
`"required"` or a specific-function choice
(forcing/restricting a call needs constrained decoding, not implemented),
`response_format` other than `{"type": "text"}` (no constrained JSON),
`logprobs`/`top_logprobs` (not returned), and `parallel_tool_calls: false`
(single-call can't be guaranteed without constraining). Unknown fields that
don't affect the output (e.g. `user`, `store`, `metadata`) are accepted and
ignored.

Non-null `thinking_token_budget`, `grammar`, `stop_regex`, `cache_salt`,
`guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `structured_outputs`,
`regex`, and `ebnf` return `400 unsupported_parameter` before generation, including
empty values. Explicit defaults `add_generation_prompt=true` and
`continue_final_message=false` are accepted; other non-null values are rejected.
In particular, `cache_salt` does not partition the cache. A caller requesting
salted isolation must not receive a successful unsalted response. Omitted/null
values are treated as unset. Harmless metadata remains accepted as above.

Non-streaming response: `chat.completion` with one `choice`
(`message.role = "assistant"`, string `content` or `tool_calls`, `finish_reason`
∈ `stop` | `length` | `tool_calls`) and a `usage` block.
When reasoning is returned, `message.reasoning_content` contains the extracted
reasoning text separately from visible `content` and `tool_calls`.
`usage.prompt_tokens_details.cached_tokens` reports prompt tokens reused from
the session's resident state or a shared prefix snapshot instead of prefilled
this request (0 when the turn fully prefilled); the streaming usage chunk
carries the same field.

Adapters with a generated-token counter also return
`usage.completion_tokens_details.reasoning_tokens`. It counts actual generated
reasoning-body token IDs, including reasoning hidden by visibility controls,
and excludes headers and delimiters. It is a breakdown of `completion_tokens`,
not an addition to the total. The same breakdown appears in streaming usage.
If exact generated IDs or the required tokenizer framing are unavailable, the
details field is omitted rather than estimated. Workers can omit replayable IDs
on stop-trimmed or cancelled turns; those turns can therefore lack the breakdown.

Streaming response: `text/event-stream` of `chat.completion.chunk` objects —
first chunk carries `delta.role = "assistant"`, subsequent chunks carry
`delta.content` (or buffered `delta.tool_calls`), a final chunk carries
`finish_reason`, optionally a usage-only chunk (with
`stream_options.include_usage`), terminated by `data: [DONE]`.
For models with an incremental parser, reasoning is emitted as
`delta.reasoning_content` while generation runs. Other reasoning extractors buffer
their output. With tools enabled, content and reasoning are buffered unless a
model adapter supplies a tool-safe incremental reasoning parser. Complete tool
calls remain buffered. Idle HTTP streams include SSE
keepalive comments every 15 seconds. Model launchers may bound initial readiness
waiting to return headers during prefill; errors detected after headers arrive
as SSE error events.

The generic and Muse batching launchers accept `--api-dialect openai|openrouter`.
The `openrouter` dialect attaches requested usage to the finish chunk, once, for
clients that stop consuming on `finish_reason`, and supplies overflow metadata
for context recovery. The default `openai` dialect retains the separate
usage-only chunk described above.

`--streaming-preflight-timeout` waits for first output, including prefill. The default
`none` preserves an indefinite wait, while finite nonnegative values allow SSE headers
before first output. Once headers are sent, later failures are SSE error events.
This bounds generation-readiness waiting, not waiting for a session transaction
lease or preparing the request before generation starts.

#### OpenRouter error contract

The compatibility dialect retains the common `error.message`, `error.type`, and
`error.code` fields. Context overflows detected by Python preflight or by an
updated native worker carry `error.metadata`. Native counts include all prompt
positions, including image positions and numeric BOS. Adapters that cannot count
all positions defer overflow checks to the worker instead of publishing a lower
bound as `prompt_tokens`. Older workers that report
only an unstructured error cannot supply recovery counts.

The multiplexed error frame uses `code: "context_length_exceeded"` and
`context_overflow: {"prompt_tokens": N, "context_length": C, "completion_tokens": M}`.
Other `invalid_argument` errors are never reclassified as context overflow.

| Key | Meaning |
| --- | --- |
| `error_type`, `provider_code` | `context_length_exceeded` |
| `context_length` | Full configured model window |
| `prompt_tokens` | Input prompt size |
| `completion_tokens` | Reserved output space: the explicit positive limit, or 1 when unset |

Let `budget = context_length - completion_tokens`. For a positive budget the
message is exactly:

```text
This model's maximum context length is {budget} tokens. However, your messages resulted in {prompt_tokens} tokens (the prompt budget reserves {completion_tokens} completion tokens from the {context_length}-token model window). Please reduce the length of the messages or completion.
```

The message exposes a usable *input budget* for clients that parse those counts;
metadata preserves the full window. The minimum one-token reservation does not
choose a client compaction margin. If the reservation consumes the whole window,
the ordinary diagnostic message is retained because input compaction cannot make
that reservation fit.

After streaming headers, this adapter sends a bare `data: {"error": {...}}` event,
then `data: [DONE]`. It does not add a `choices` entry or `finish_reason: "error"`.
This is the adapter's supported error contract in both dialects; the dialect
selects overflow metadata and usage placement, not the full OpenRouter API.

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

For named sessions, an unchanged assistant reply with the same preceding message
history can reuse its original generated token IDs. Compaction, removal, or edits
to earlier message content invalidate affected records, even if the assistant's
visible answer is identical. Clients may omit `reasoning_content` or send `null` when echoing a reply;
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
selection, prompt-boundary caching, and native-backed Python integration checks.

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
of strings. `ignore_eos` is a JSON boolean, default `false`; it suppresses
service-default stop tokens while preserving explicit stop strings. Unknown
fields and segment types, including images or image payload fields on otherwise
valid text segments, fail explicitly instead of silently dropping modalities.

Output JSON strings replace invalid/incomplete UTF-8 with U+FFFD, matching the
legacy worker's serialization policy; a generation ending mid-character does
not fail unrelated requests.

Optional `generated_token_ids` in a terminal is present only when the runtime can
safely replay exact output IDs. Statistics describe consumed prompt and generated
tokens, not necessarily visible text. Runtime failures have the terminal shape
`{"request_id":8,"error":"message","code":"internal"}`. Rejections before
runtime admission additionally include `"admitted":false`; clients must not infer
that a named session was created from such a rejection. An absent `admitted`
field leaves admission unknown. Stable error codes
include `invalid_argument`, `not_ready`, `session_not_found`, `session_busy`,
`capacity_exhausted`, `session_capacity_exhausted`, `internal`, `slow_consumer`, and `frame_too_large`.

Lifecycle commands are `open`, `close`, and `reset`, each with `session_id` and
its own request ID. These commands do not block the input reader.
Their replies are respectively `opened:true`, `closed:true`,
and `reset:true`, or a correlated error. Close remains idempotent and includes
`existed:true` when it released a logical session slot, or `existed:false` when
the session was already absent. Older workers can omit this field; a client
cannot then distinguish stale candidates from released slots. Older multiplexed
workers also use `capacity_exhausted` for session-slot refusals, so automatic
eviction does not trigger for those workers. The legacy `WorkerClient` maps its
explicit session-open capacity refusal to session-slot exhaustion; its close
reply may still lack existence information.
An accepted close/reset fences its session: earlier generation text and terminals
precede its ACK, and runtime-owned generation callbacks and captures are cleaned
up before the ACK is enqueued. Later accepted same-session operations stay behind
that ACK, including commands submitted before it. Other sessions can progress.
Immediate validation/admission rejections and cancel replies are not covered by
this ordering. Close acknowledges logical release, not physical executor cleanup.
Open creates an empty session or preserves an already-open session. Reset
acknowledges discarding live state and history and reopening the same key; a
failed replacement does not restore the old state. Matching shared prefix
snapshots remain eligible for subsequent generation from its supplied full
prompt, so open/reset does not guarantee a cold prefill.

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
`--prefix_cache_entries=N` enables retained-prefix reuse for both greedy and
sampled requests. With `--prefix_capture_slots=C`, it adds `N + C` physical rows
for retained snapshots and transient captures; logical `--max_sessions` remains
unchanged. The default `--prefix_capture_slots=0` resolves `C` automatically to
`min(max_sessions, max_inflight_requests)`, so overlapping requests can each capture
their prompt. `--prefix_checkpoint_prompt_end=initial` (the default) captures the
prompt end of each request that creates a session, including every anonymous
request but not the first generation after open or reset.
`--prefix_checkpoint_prompt_end=every` captures the prompt end of every request
with the same concurrent allowance; it requires nonzero cache capacity. Without
periodic checkpoints, a prompt end already in the cache is
refreshed instead of cloned again, and a snapshot is published and its transient
slot released at the source request's first output after the clone completes, or
when the request finishes.
With periodic checkpoints enabled, automatic sizing uses at most two
rows because checkpoint collection remains serialized and one replacement can
overlap. A positive value overrides automatic sizing; values above two add unused
row capacity in periodic mode. Disabled prefix caching reserves no capture rows.
Capture is best effort. `--prefix_checkpoint_prompt_end=every` cannot be combined
with a nonzero checkpoint interval or a custom checkpoint policy; invalid
combinations are rejected. A C++ `prefix_checkpoint_policy` overrides the
interval and prompt-end capture.
Periodic/custom checkpoints share a serialized lane; a pending clone can skip
another boundary. See the [batching-backed serving guide](batching.md) for what
`every` reuses in practice and its MLX memory cost.

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

Idle-session recovery retries at most once and only for native
`session_capacity_exhausted` (logical session slots), or a legacy named-session
open refusal. Request, control-queue, executor, and output-buffer overload remain
`capacity_exhausted` and never trigger eviction. Both map to HTTP 429. Failed
closes retain replay state and rotate the victim for a later request.

Unconfirmed native admissions whose cleanup close fails remain cleanup candidates
until closure is confirmed. Updated workers report whether close released a slot,
so stale candidates do not consume the admission retry. Native front-door and
local request-limit rejections carry `admitted:false`; they cannot have created
a session and do not need a cleanup close. Other errors without output can leave
session state, so they still require cleanup. Eviction skips sessions whose
runtime lease is still settling, and disconnected consumers defer their cleanup
rather than waiting for that lease. Task cancellation prevents an admission retry,
even when it arrives during generation cleanup.

Client-only `cache_control` markers do not change the replay history fingerprint.
A string and its equivalent single text-part content are also interchangeable.
Worker completion metadata includes `matched_stop`, the exact string removed
from generated text, or `null` when no string stop matched (including token EOS).
An omitted key identifies a legacy worker whose string-stop cause is unknown.
Tool adapters use this distinction to separate request-stop truncation from
malformed completed model output. Request stops must precede model stops in the
worker's stop list: matching uses the earliest text position, then the earliest
list entry to break ties, so request stops win at the same position.

Tool-definition changes are rendered into the next full prompt and checked by the
native prefix comparison. They do not invalidate an unchanged historical assistant
echo: stored token IDs still describe that past output, while changed prompt
prefixes require fresh prefill.

`reasoning.exclude` and `include_reasoning` control reasoning visibility in both API dialects. With `ignore_eos`, completion usage still counts every generated token, but reasoning-token details are omitted when the visible span cannot be aligned safely.
