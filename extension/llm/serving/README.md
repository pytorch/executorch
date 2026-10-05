# Serving Runtime

`ServingRuntime` admits full prompts and emits ordered text events followed by
one terminal event. Source segments preserve order: text is tokenized separately,
token IDs are retained verbatim, and opt-in encoded images are CPU-preprocessed
into owned `Image` values. Raw decoded-image, audio, and video prompt segments are
not accepted by this public serving path.

Every request prepares its complete prompt through `Runner::prepare_async`,
including text-only requests and cache hits. Control retains the logical session
claim while the engine prepares an immutable executor-private payload. Completion
returns through the existing control condition variable; serving never waits on
a preparation future or inspects the private payload. Decoder opening, cold
replacement, prefix selection, and generation follow successful preparation.
Preparation failure leaves existing decoder state and logical history intact.

## Images and Bounds

Image support requires `ServingRuntimeConfig::max_images = 1`, a positive fixed
`max_image_preprocessed_bytes` bound (default zero disables images), an
`image_preprocessor` callback, and an executor whose fixed `PreparationConfig`
advertises image support. The image-output bound plus minimum input/container
overhead must fit executor workspace. `ServingInfo` reports zero image limits
otherwise; invalid or unfit image settings do not disable text serving.
Supported source MIME types are `image/png` and `image/jpeg`. Limits can be lowered
from the hard maxima: one image, 512 KiB encoded bytes, 4096 pixels per dimension,
and 4 Mi pixels. Allocation-free PNG IHDR/JPEG SOF preflight checks source bounds
before invoking the CPU callback; it does not validate CRCs or compressed pixels.

Before invoking the callback, serving stages all tokenized text/ID segments,
including those after the image, and an empty `Image` placeholder in the final
segment container. `PreparationConfig::input_retained_bytes` measures actual
staged capacities (`T`). The declared pixel-buffer capacity bound (`B`) must
satisfy `B <= W - T`, where `W` is executor workspace. After decoding, serving
fills the placeholder in place and checks returned capacity bytes against `B`,
as well as dimensions and storage size.

The callback is trusted implementation code running on control. It must capture
and enforce `max_image_preprocessed_bytes` before allocating output, validate
the complete encoded format, return an owned CHW image with one to four channels,
and make no model calls. Codec scratch has a separate trusted bound of
`max_image_pixels * 4 * sizeof(float)`. The unchanged callback signature carries
no per-call preparation limits. This contract is not a sandbox for arbitrary
callback allocations. Tests use a synthetic CPU hook and executor-private
payload, not a production image codec or model integration.

## Reuse and Lifetime

Full text-token identity is separate from prepared backing. Creation-only prefix
reuse selects a range of the full prepared owner; exact hits retain the last
prompt token for a fresh forward pass. Incoming or resident image history always
cold-replays and bypasses both token-cache lookup/capture and warm text-history
reconciliation, even if images expand to equal position counts.

`max_prepared_bytes` reserves the sum of the executor's fixed workspace/input
and output bounds before CPU preparation, retaining the conservative charge
through terminal callback/capture cleanup. The image-output bound is covered by
that existing workspace reservation, with no extra counter or double charge.
Raw input includes segment-container and image/token storage capacity; Runner
rejects inputs above the workspace bound before queueing. Its own aggregate
accounting independently covers queued
sources and prepared backing. Transient codec scratch is separately bounded by
the trusted CPU hook contract. This is not a process-wide RAM limit: model
weights, decoder caches, and incoming request bodies are outside this budget.
Cancellation, reset, close, and shutdown still fence earlier same-key callbacks
and capture destruction. A request cancelled during preparation cannot
open its generation's decoder session after preparation completes.

`prompt_tokens` counts only text tokenizer/ID tokens. `prompt_positions`,
`reused_prompt_positions`, and `prefilled_prompt_positions` account for decoder
positions, including image expansion. Token reuse/prefill counters remain zero
for image prompts because opaque position ranges do not expose a token mapping.
Generated token IDs remain available for successful replayable text output.
