# Batched generation, session cloning, and prefix reuse

`Runner` schedules generation inputs through an `Executor`. `ModuleExecutor`
runs raw-token batches against a registered off-graph KV cache. Session cloning
is a runtime operation; token matching and snapshot eviction are a separate,
caller-owned policy in `batching::PrefixCache`.

## Opaque inputs

Ordinary text and decode use raw token IDs. Supporting executors can also consume
an opaque `PreparedInputPtr` through `session.generate_async(input, config,
callback)`. Its model-specific representation must be compatible with that
executor; the framework provides no preparation or construction API.
Executors opt in through `Executor::accepts()`, which defaults to false. The runner
checks it before scheduling, so rejected opaque input fails only its generation.
`PreparedInput::kind()` provides a process-local type identity for this check,
without RTTI; matching a type does not establish model or device compatibility.

The complete backing reports a fixed decoder-position count through `size()`.
Prefill chunks share its owner and slice its logical positions; decode feedback
remains raw tokens. Input metrics count positions, not necessarily text tokens.
Metadata and logical contents stay immutable. `ModuleExecutor` remains raw-token-only.

## Cloning a session

`Session::clone_async(upto)` returns a future containing a new Session, or
`nullopt` when cloning is unsupported, the boundary is unavailable, resources
are exhausted, or the runner is stopping. `upto` is the exclusive end of the
committed prefix `[0, upto)`. The child starts idle at exactly that position,
with independent write ownership and no pending prediction, active generation,
callback, or inherited sampling state. Its next generation sets its sampler.

The request runs between forwards on the engine thread:

```
Session::clone_async(upto)
  -> Runner command
  -> Executor::clone(source, upto)
  -> ModuleExecutor
  -> BatchControl::seq_clone(source_sequence, upto)
```

The runner checks the accepted committed boundary and session validity;
ModuleExecutor also checks the returned backend position. A speculative
executor can have physically written beyond accepted output, so physical
cache length alone does not authorize a clone. The executor must either
return the exact requested prefix or refuse without changing the source.

A clone can be requested while its source is generating. An admitted clone
ordered before source closure remains eligible even if the caller destroys
the source before the command runs. Shutdown can refuse the request, but must
settle its future and release any unpublished child. Dropping a clone future
also releases a successful child when its last owner goes away.

Callbacks may enqueue cloning but must never wait for its future. Commands
queued synchronously by an output callback run before the next forward. This
ordering allows a prompt snapshot to be captured before decode overwrites
sliding-window state.

## Prefix policy

Create `PrefixCache(max_entries)` outside the Runner to store reusable snapshots.
`lookup()` returns an independent session and the number of reusable tokens;
submit the remaining suffix, or start a fresh session on a miss. Keys must
identify the complete prompt under the same model and execution configuration.

Capture the prompt before decode, then collect it after generation:

```cpp
auto capture = prefixes.capture_prompt(session, prompt);
auto generation = session.generate_async(
    suffix, config, capture.wrap(on_update));
generation.wait();
capture.collect();
```

Serialize cache operations on the caller thread. `lookup()` and `collect()` may
block and must not run in Runner callbacks. Keep the source session alive until
generation completes and the cache alive through collection.

For intermediate snapshots, use `GenConfig::prefill_breakpoints` and
`PromptCapture::observer`; see [runner.h](runner.h) and [prefix_cache.h](prefix_cache.h).
Capture is best effort and requires executor support at the requested boundary.

Default retention is bounded global LRU, including periodic checkpoints; early
snapshots are not protected. Pass a custom policy to `PrefixCache` to change
retention. Checkpoints can carry application-defined metadata such as priority
or boundary class. See [prefix_cache_retention.h](prefix_cache_retention.h) for
the policy interface and default implementation.

Reuse can change floating-point results as prefill shapes change. Validate
numerical behavior for your model and backend.

## Sliding windows and backends

The backend must preserve every layer's history needed to continue at the
requested position, or refuse cloning. A token match alone cannot establish
that the required K/V state is still resident. The prefix manager neither
computes sliding windows nor bypasses retention checks.

* `CellCache` shares prefix cells by ownership bits. It retains full history;
  sliding attention restricts the mask rather than evicting owned cells.
* `MLXBatchedSequenceCache` owns a `SequenceCache` per sequence. Ring layers
  evict physical rows. Its clone checks every layer's rewind restriction and
  written high-water mark; rewinding does not lower that high-water mark.

Snapshotting complete prompts does not retain every earlier branch point.
A shared prefix outside every snapshot's retained window is a safe cache miss.
The backend owns tensor allocation, stream ordering, and write isolation.
It may copy K/V or share storage, so compute reuse does not imply proportional
memory savings.

Backends using `ModuleExecutor` provide a registered cache builder,
`BatchControl`, and storage compatible with the clone and rewind contracts.

## Ownership and capacity

The `max_sessions` argument to `ModuleExecutor::create` limits **all resident sessions**:
active requests, retained snapshots, and in-flight clones. The caller must
reserve room for all three. Each session contributes `max_session_tokens` to
the logical capacity bound, with overflow and backend sequence limits checked
at creation. This is not a byte budget; storage sharing is backend-dependent.

Destroying a Session queues its close. Subsequent opens/clones are ordered
after that close; `Runner::shutdown()` completes cleanup. Keep the Executor
alive through runner shutdown. Keeping PrefixCache outside Runner also avoids
an ownership cycle: its Sessions keep runner internals alive.

## MLX example and validation

`mlx_run_llm_batched` exposes `--prefix_cache_entries` (default zero) and
`--prompt_rounds` (default one). It prepares each round through PrefixCache,
wraps callbacks with `capture_prompt()`, then collects snapshots on the caller
thread after generation completes.
With P prompts and E cache entries, it reserves `P + E + min(P, E)` resident
slots: P requests, E retained entries, and at most min(P, E) pending snapshots.
Only those first min(P, E) prompts are selected for snapshot capture per round.

```bash
mlx_run_llm_batched \
  --pte model.pte --tokenizer tokenizer.json \
  --cache_kind batched-sequence \
  --prefix_cache_entries 4 --prompt_rounds 2 \
  --max_decode_sequences 1 --temperature 0 --seed 42 \
  --out_prefix cached "Explain sliding-window attention."
```

Use a model exported with off-graph cache and full or selected logits. With
one prompt, compare `cached_0.txt` and `cached_1.txt`; the second request should
report reuse of all but the final prompt token. Prompts prepared before a
snapshot is inserted can all miss.

`extension_llm_batching_test` covers general clone lifecycle, callback ordering,
pending/speculative boundaries, token matching, LRU, ownership, refusal, and
suffix execution. `extension_llm_cache_test` checks direct cloning with real
SequenceCache plans and physical CPU K/V arrays against cold attention across
ring wraps and branches, including an intentionally unsafe rewind control.
MLX tests exercise direct clones against real attention/storage.

```bash
cmake --preset mlx-release -DEXECUTORCH_BUILD_TESTS=ON
cmake --build cmake-out --config Release --target mlx_batched_sequence_cache_test
ctest --test-dir cmake-out -C Release -R '^mlx_batched_sequence_cache_test$' \
  --output-on-failure
```

The common tests run on Linux/macOS CMake CI; the dedicated MLX workflow builds
and runs the MLX batched-cache target. See the
[MLX smoke test](../../../backends/mlx/examples/llm/README.md#prefix-cache-smoke-test)
for real-model validation. A Mac is needed for MLX execution, not for testing
the shared session and prefix policy.
