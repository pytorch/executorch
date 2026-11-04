# Muse Glimmer Batching Serving

The batching runner and multiplexed worker support MLX and CUDA autoregressive
generation with text only or text plus one image across the submitted
conversation. They are opt-in entry points; the
legacy runners and export defaults remain unchanged. DFlash batching runs on MLX
(text only) and CUDA, which also serves images when exported with `--mmproj`.
CUDA specifics are in [CUDA](#cuda) below.

## Export

Export with the off-graph cache ABI required by the batching runner. Include the
vision projector for image support. Replace the placeholder paths with your local
checkpoint and output paths.

```sh
python -m executorch.examples.models.muse_glimmer.export.export_solo \
  --backend mlx \
  --gguf /path/to/models/mg/model.gguf \
  --mmproj /path/to/models/mg/projector.gguf \
  --use-offgraph-kv-cache \
  --activation-dtype float16 \
  --max-prefill-chunk 512 \
  --max-vision-patches 4096 \
  --output-dir /path/to/models/mg-batching
```

This artifact supports packed batching execution and is not interchangeable with
an artifact exported for the legacy runners.

Off-graph MLX export uses the checkpoint's native context limit and ignores
`--max-seq-len`, which remains a legacy export override. Nondefault values emit a
warning before checkpoint loading. Missing or invalid native context metadata is
an error. The three relevant limits are independent:

- The checkpoint's native context limit bounds token positions.
- `--max-prefill-chunk` bounds positions processed per model forward.
- The server's `--max-context` controls the runtime context/cache budget and may be
  lower than the exported limit. Exporting the full context does not allocate that
  full KV capacity at runtime.

The DFlash exporter accepts `--backend mlx --use-offgraph-kv-cache`.
Serve that artifact with `--dflash`. Draft proposals are always greedy;
temperature, top-p, and top-k apply only to the target distribution.
Set `--dflash-n-draft N` at startup to propose N tokens using block size N+1.
Omitted or zero uses the exported maximum minus one; larger values are rejected.
Re-export older fixed-block artifacts before using this option. One updated
artifact supports different draft counts without another export. Draft tokens
have separate `[owners, block_size]` axes; each block ranges from 2 to the
exported maximum. Torch infers the concatenated context-plus-block K/V length.

## Build

On Apple Silicon, run the Make target from the ExecuTorch checkout:

```sh
cd /path/to/executorch
make muse-glimmer-batching-mlx
```

This builds and installs the MLX dependencies, then builds
`muse_glimmer_batching_runner` and `muse_glimmer_batching_worker` in
`/path/to/executorch/cmake-out/examples/models/muse-glimmer-batching`. It does not
export a model or run tests. Use `make muse-glimmer-mlx` for the legacy runners.

### Custom CMake Builds

The Make target handles the normal configuration. For a custom build, enable
`MUSE_GLIMMER_BUILD_BATCHING` in the MG example and use an MLX-enabled ExecuTorch
installation with `EXECUTORCH_BUILD_EXTENSION_LLM` and
`EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER` enabled. Set `EXECUTORCH_BUILD_MLX=ON` and
`EXECUTORCH_BUILD_CUDA=OFF` in the example configuration as well.

Optional overrides are `MUSE_GLIMMER_STB_INCLUDE_DIR` for local stb headers and
`MLX_METALLIB_PATH` when the installed package does not supply the metallib.
Preparation/materialization tests can be enabled with
`MUSE_GLIMMER_BUILD_BATCHING_TESTS=ON`, built as `muse_glimmer_batching_test`, and
run with CTest's `muse_glimmer_batching` test. Both MG build options default to
`OFF`; the tests can be built without the batching executor but still require the
MLX configuration.

## Serve

Use the worker built by the Make target and the exported model artifacts:

```sh
python -m executorch.examples.models.muse_glimmer.serving.serve_batching \
  --worker-bin /path/to/executorch/cmake-out/examples/models/muse-glimmer-batching/muse_glimmer_batching_worker \
  --model-path /path/to/models/mg-batching/model.pte \
  --data-path /path/to/models/mg-batching/model.ptd \
  --pos-embed-path /path/to/models/mg-batching/pos_embed.bin \
  --tokenizer-path /path/to/models/mg/tokenizer.json \
  --hf-tokenizer /path/to/models/mg \
  --backend mlx \
  --max-context 16384
```

Omit `--data-path` when weights are embedded in the program. For text-only
artifacts, omit `--pos-embed-path`. The launcher uses one multiplexed worker
process and the bundled Muse chat template with the Hugging Face tokenizer. The
template preserves developer messages and assistant commentary with tool calls.
It does not fall back to a
legacy worker or DFlash.

Send `"stream": true` for streaming responses. The MG parser removes model
framing and emits `reasoning_content` and `content` incrementally. Optional ATEM
tool parsing is enabled with `--tool-parser atem`. Reasoning preceding tool calls
streams incrementally; complete calls are buffered and validated before delivery.
Malformed parameter bodies, duplicate parameters, undefined calls, and arguments
that violate their JSON Schema produce `invalid_tool_output` errors, with no tool
calls emitted. Parameters support union schemas and local JSON pointer references;
string values preserve whitespace and literal Harmony headers. Schema validation
accepts Draft 7, 2019-09, and 2020-12 (the default), including root `$id` and local
fragment references. External references, nested schema identifiers/dialect changes,
dynamic references, more than 1023 coercion alternatives, and more than 4096 expanded
schema nodes are rejected before generation with HTTP 400 `invalid_tool_schema`.
All active tool schemas must pass validation, including tools with `strict=false`
and tools the model does not call. Malformed schemas (for example, a property with
`"required": true`), Draft 4 or 6, and unknown `$schema` dialects are rejected.
Disabled tools (`tool_choice="none"`) are not validated.
Validators are compiled once per request and never fetch schemas. Install the
serving requirements, including `jsonschema` and `regex`.
Schemas without `$schema` use Draft 2020-12; legacy tuple `items` need an explicit
Draft 7 declaration, and Draft 4 boolean `exclusiveMinimum` is unsupported.
Undefined tools, invalid arguments, and exhausted output budgets return HTTP 502
`invalid_tool_output` (or an SSE error after streaming starts). SDKs may retry
these responses; clients wanting a single attempt should disable automatic retries.
Coercion preserves correlations between object alternatives and conditional schemas,
then validates the complete argument object. Declared numeric parameters also accept
spellings such as `+5`, `007`, `.5`, and `5.`; declared strings stay literal.
Before generation, the server walks every coercion alternative and declared parameter,
including map-property projections. All active tools share a 65,536-visit preflight
budget. Requests exceeding that budget return HTTP 400 `invalid_tool_schema`. The output budget also
counts actual keyword checks, nested values, and deep comparison work, and is shared across all invocations
in a response. Client `pattern` and `patternProperties` searches share a separate
100 ms regex execution budget, including annotation and property-projection searches.
Patterns are compiled and evaluated with the same timeout-capable `regex` engine,
including Unicode properties and named groups. `uniqueItems` uses bounded structural
keys rather than pairwise comparisons of objects; `enum` and `const` share this budget.
Extra parameters, large nested arguments, or multiple calls can exhaust this budget
and return `invalid_tool_output`; coercion tries at most 1024 alternatives per call.
This is validation after generation, not token-level constrained decoding.
`function.strict=true` is supported: a successful response contains only
schema-valid calls; invalid generation fails the response. This supports Muse's
strict summarizer tool. Required/named tool choice and structured response formats
remain unsupported.

The launcher waits for initial generation readiness before returning streaming
headers. Set `--streaming-preflight-timeout 1` to bound generation-readiness
waiting, including prefill, to one second. This does not bound earlier request
preparation or waiting behind another request on the same session.
Idle streams send SSE keepalive comments every 15 seconds. Admission errors
detected before headers retain their HTTP status; later session-slot rejection
before any raw model output can evict an idle session and retry, including after
headers. Other late errors are delivered as SSE errors. Keepalives indicate a live connection, not
measured prefill progress.

### OpenRouter compatibility

Start the server with `--api-dialect openrouter` for OpenRouter clients and
`--tool-parser atem` for tool calling. `/v1/models` reports the configured
`context_length`; clients must consume that value or configure their context
budget explicitly. The server does not configure client-side compaction.

The OpenRouter usage format includes usage once on the terminal finish chunk.
The installed Muse provider stops reading at that chunk, so a later usage-only
chunk would be lost. The default `openai` format keeps the separate usage-only
chunk. Both formats honor `stream_options.include_usage`.
The `openrouter` mode also supplies the error metadata and message counts Muse
needs for reactive context-overflow recovery, including worker-detected overflow
when using an updated native worker. Worker counts include image positions and
numeric BOS. Image prompts bypass Python token-count preflight so only the
worker publishes their image-expanded count; older unstructured worker errors
cannot provide recovery counts.
Counts describe the input prompt and
its available budget after reserving completion space; metadata retains the full
model window. A completion reservation consuming the entire window cannot be
fixed by compacting input: diagnostic metadata remains, but the message does not
match Muse's recoverable prompt-budget format. With no explicit `max_tokens`,
the server reserves one output token; Muse applies its own compaction margin.

The model adapter qualifies each declared flat tool name as `name.name` in the
prompt and maps that exact alias back to the original name in responses. Dotted
names stay unchanged; colliding aliases are rejected. This mapping is local to
the Muse model adapter and only accepts declared tools. OpenRouter
`reasoning: {"effort": "low"}` and `reasoning_effort` map to the template's
`reasoning_strength`; supported values are `low`, `medium`, and `high`.

Streaming intentionally differs from the legacy buffered fallback for incomplete
`<|` headers, which are suppressed at EOF; bare `to=` suffixes remain literal text
unless a complete addressed header resolves them. Streaming also removes known
control tokens inside reasoning and omits separators for control-only bodies.

Image requests use the OpenAI `image_url` content part with an inline base64 JPEG
or PNG data URL. Remote URLs, file paths, `detail`, and more than one image across
the entire history are rejected. Resubmit the image inline when it is part of the
submitted history; server-side image references are not supported.

### Limits

- `--max-sessions`, `--max-inflight-requests`, and `--max-decode-sequences` default
  to 4. They control session capacity, request admission, and decode batch size
  separately. Set all three according to available memory and desired concurrency.
- `--max-inflight-requests` accepts positive signed-32-bit integers without a
  build-time concurrency ceiling. The worker advertises its capacity to the client.
- `--max-context` must be greater than 1, fit a signed 32-bit integer, and be
  supported by the exported model.
- `--max-vision-patches` defaults to 4096, versus the export CLI default of 16384.
  It accepts 4 through 2147483647; the worker clips to the artifact's capacity and
  complete four-patch groups. Higher caps can improve image detail at the cost of
  more vision compute and memory; lower caps can reduce image quality.
- `--max-image-bytes` is 1 byte through 20 MiB (default 20 MiB), measured on the
  compressed image after base64 decoding, not on decoded pixels.
- `--max-request-bytes` is positive and at most 32 MiB (default 32 MiB), including
  JSONL framing and the newline. It must be at least
  `4 * ceil(max_image_bytes / 3) + 1024`; lower both limits together when needed.
- Output frames have a separate fixed 1 MiB limit. BOS/EOS IDs must fit unsigned
  64-bit integers.

The base64/framing check does not guarantee that an arbitrarily long text history
fits. Whole-frame, decoded-pixel, vision-patch, and model context limits still
apply. Text/image prefix snapshots are opt-in via `--prefix-cache-entries`
(default 0).

## Session Behavior

Prefix caching retains the initial prompt by default. Add
`--prefix-capture-prompt-end` with `--prefix-cache-entries` to also retain the
prompt boundary of subsequent requests, including tool-result continuations
and replacements. Repeated or branched requests can then reuse those snapshots.
New tool-result text still needs prefill. This is best-effort capture within the
same bounded cache and transient-row limits, with global LRU retention; it can
evict earlier prompts. Benchmark capture cost and memory for your workload
before enabling it. It also works with `--prefix-checkpoint-interval` and adds
the exact prompt end when it falls between periodic checkpoints. A pending
checkpoint or another request can occupy the capture lane and skip a boundary;
use prompt-end capture without periodic checkpoints when turn reuse is the
priority. Automatic transient-row sizing is capped at two in either mode. The C++
`prefix_checkpoint_policy`, when supplied, overrides both options.

- Named sessions persist by default. `--evict-idle-sessions` allows the server to
  close the least recently used idle session when logical session slots are exhausted, clearing
  its transcript cache as well. Active requests and queued users of a session are
  protected. Resubmitting an evicted session with full history recreates it.
  This opt-in policy also allows anonymous requests to evict named idle sessions;
  the anonymous request releases its temporary slot when it finishes. If all
  slots are active, capacity errors still apply. Request/queue overload does not
  trigger eviction. Streaming and non-streaming requests both confirm a new
  named session on the first raw output or successful terminal response.
- Generated-token replay is bound to preceding history as well as the assistant
  response. Compaction or edited earlier messages invalidate affected records,
  preventing an identical visible answer from restoring another turn's reasoning.
- Text and image prompts use the shared prefix-reuse path. Image rows count as
  decoder positions; matching uses token IDs and an image-content digest with
  content-relative row offsets, not patch token IDs alone.
- Resubmit the complete history, including the image, on every turn. A clean
  named session whose history is an exact prefix reuses its KV and executes only
  the remaining prepared suffix. Pending generated tokens are handled by Runner.
- Opt-in snapshots can also reuse mixed prefixes across requests. Existing
  policies remain: equal, shorter, mismatched, or dirty named histories replay;
  snapshots leave at least one fresh prompt position to execute.
- Omitting an earlier image changes the supplied context; serving does not
  implicitly retain omitted image or text history.
- Metadata, options, and prompt preparation are validated before context
  replacement. A later engine or vision failure does not restore replaced state
  and follows the shared batching failure contract: a runtime vision-encode error
  fails every request in that execution batch, including text-only requests.
- Vision encoding runs inside `execute()`. Cancelling an image request does not
  interrupt that encoding or immediately unblock co-batched text requests.

## Implementation

`MuseGlimmerMLXExecutor` implements `batching::Executor` directly and owns model
execution, MLX cache/sequence state, packing, sampling, and embedding
materialization. The worker reuses the shared batching, serving, and multiplexed
transport infrastructure.

CPU source decoding and model preparation run after admission on the control
thread. The source adapter produces ordered `PromptInput` segments; the
runtime-owned MG model preparer tokenizes text, expands image rows, and supplies
prepared backing and prefix identity. Image identity hashes decoded RGB,
dimensions, and the computed vision grid within one immutable model runtime.

Vision encoding, text embeddings, and decoder calls run on the engine thread,
so vision encoding delays other co-batched requests. Prepared image inputs are
validated once at construction. Suffix views share the same backing and lazy
image embeddings, including views created after encoding. Decoded RGB is released
after validated embeddings are cached; failed encoding retains it for retry.

KV-prefix reuse is not a cross-request image-embedding cache. Independently
prepared requests retain independent backing. If the uncached suffix skips the
image entirely, it needs no vision encoding; if it intersects the image, that
backing may still encode the full image before selecting the remaining rows.
Prefix identity does not retain RGB or embedding buffers.

## CUDA

The CUDA worker builds `CudaExecutor` through `CudaExecutorConfig` under the same
serving runtime, prefix cache, scheduler and launcher. Export with the off-graph
cache, which on CUDA writes the batching methods: static `forward_{N}` (each
captured into a CUDA graph), a dynamic `forward_others` for wider steps, and
device samplers. Without `--use-offgraph-kv-cache` CUDA keeps its in-graph
prefill and decode export.

```sh
python -m executorch.examples.models.muse_glimmer.export.export_solo \
  --backend cuda \
  --gguf /path/to/models/mg/model.gguf \
  --use-offgraph-kv-cache \
  --output-dir /path/to/models/mg-batching-cuda
```

`--forward-sizes` (default `1,2,4,8,16,32,64`), `--max-step-tokens` (default
4096) and `--max-cells` (default 65536) size the methods and the shared KV
pool; `--logits` samples on the host instead. Add `--mmproj` for images: the
token forwards are unchanged, and the artifact gains `vision_encoder`,
`embed_text` and a dynamic `forward_from_embeddings` over the same cache, which
runs only the prefill steps that carry an image.

Build with `make muse-glimmer-batching-cuda` on Linux, then serve:

```sh
python -m executorch.examples.models.muse_glimmer.serving.serve_batching \
  --backend cuda \
  --worker-bin /path/to/executorch/cmake-out/examples/models/muse-glimmer-batching/muse_glimmer_batching_worker \
  --model-path /path/to/models/mg-batching-cuda/model.pte \
  --data-path /path/to/models/mg-batching-cuda/aoti_cuda_blob.ptd \
  --tokenizer-path /path/to/models/mg/tokenizer.json \
  --hf-tokenizer /path/to/models/mg \
  --max-context 8192
```

`--data-path` is required. Image artifacts read `pos_embed.bin` beside
`model.pte`, or `--pos-embed-path`. Every resident session reserves `--max-context` cells of the
pool: requests, `--prefix-cache-entries` snapshots, and the resolved number of
capture slots when snapshots are enabled. `--prefix-capture-slots 0` (the default)
matches request concurrency, capped at two when checkpoints are enabled; a
positive value sets an explicit limit. Their total reservation must leave one
cell for padded writes within the exported `--max-cells`. More capture slots
reduce the context available per session in the native worker's automatic
context mode. This Python launcher requires an explicit `--max-context`;
reduce the context or slot counts if they do not fit.
The cell pool tracks at most 64 resident sessions. Prefix snapshots clone a
session's cells by reference, sharing the prompt's KV rather than copying it.

For DFlash, export the tapped target and packed draft with
`export_dflash --backend cuda --use-offgraph-kv-cache` (add `--draft-max-cells`
and `--max-draft-owners`), then add `--dflash` to the serve command above. The
draft ABI and `--dflash-n-draft` behave as on MLX. Drafts are greedy; the target
accepts or resamples them on the device under each request's sampling. Decode
drafts of up to four owners run in CUDA graphs.

ATEM tool intent must start a line outside Markdown code or immediately follow a
tool recipient header. User-directed examples and quoted markers remain content.
Once a recipient is established, a subsequent header changes the recipient only
after `<|eom|>` or `<|eot|>` and optional whitespace. A quoted header inside an
ongoing message remains literal content, including when a real tool call follows
the terminated message. Fenced examples retain their controls, including end
markers. An unclosed fence keeps subsequent headers literal; a tool call after
such a fence is deliberately not executed. Unexpected text between an end marker
and the next header also keeps the existing recipient, so text after a malformed
self-to-user transition may remain hidden reasoning. Conversely, a rejected
`to=self` header inside a visible message stays literal visible text, even when
reasoning is hidden: visibility controls filter accepted reasoning channels,
not header-like strings quoted in ordinary content. The rule applies to buffered
and streamed responses.
An invocation truncated by length or a request stop suppresses the entire
incomplete call set and ends with `finish_reason=length` or `stop`, respectively;
malformed completed calls still return
`invalid_tool_output`. Streaming may already have delivered reasoning before that
error. Tool calls are emitted only after the whole call set passes validation.

Empty reasoning options, null `exclude`/`enabled`, and `enabled=true` are no-ops. `effort` accepts low,
medium, or high, and `exclude` controls response visibility. Disabling reasoning
generation or unsupported reasoning keys returns an explicit parameter error.
`include_reasoning` is a visibility alias available with both `--api-dialect`
settings; contradictory visibility controls fail before generation. Reasoning
usage is reported as `completion_tokens_details.reasoning_tokens` from exact
generated IDs even when reasoning is hidden. When the worker omits those IDs
(for example, a stop-trimmed turn), the breakdown is omitted.

The shared API rejects non-null `thinking_token_budget`, `grammar`, `stop_regex`,
`cache_salt`, vLLM `guided_json`/`guided_regex`/`guided_choice`/`guided_grammar`/
`structured_outputs`, and SGLang `regex`/`ebnf` with `400 unsupported_parameter`.
Explicit defaults `add_generation_prompt=true` and `continue_final_message=false`
are accepted; other non-null values are rejected. Cache partitioning by salt is
not implemented.

Native workers always report `matched_stop`: a removed string stop or null for
EOS/length/cancellation. Request stops take precedence over template stops at the
same position. Older workers that omit the field cannot distinguish EOS from a
removed request stop; with request stops configured, incomplete blocks are
conservatively treated as cutoffs. Completed malformed blocks still fail.

Fence delimiters are checked as complete lines: backtick opening info strings
cannot contain backticks, and closing runs allow only trailing whitespace. A
Harmony message terminator (`<|eom|>` or `<|eot|>`) can also end a delimiter line.
Streaming waits for a pending delimiter line to resolve before interpreting its
headers. ATEM parameter values retain their literal controls and headers even
when tools are disabled. Native EOS and explicit request stops still terminate
generation, including inside examples. Valid unfinished ATEM names and parameter
values remain buffered until their delimiters arrive or generation ends; their
accumulated prefixes are not rescanned. Bare tag mentions that cannot match an
ATEM opening tag are emitted as ordinary text.
With `ignore_eos`, reasoning-token details are omitted
because the generated IDs can contain hidden post-EOS text; total completion
usage still includes all generated tokens.
