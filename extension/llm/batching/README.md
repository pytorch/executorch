# Batched generation, session cloning, and prefix reuse

`Runner` schedules token deltas through an `Executor`. `ModuleExecutor` runs
those batches against a registered off-graph KV cache. Session cloning is a
runtime operation; token matching and snapshot eviction are a separate,
caller-owned policy in `batching::PrefixCache`.

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

Include `extension/llm/batching/prefix_cache.h` and construct
`PrefixCache(max_entries)` outside the Runner. It owns idle snapshot Sessions
and a bounded LRU of their exact token histories. Serialize its calls on the
application thread; `lookup()` and `PromptCapture::collect()` can wait for clone
completion and cannot run in a Runner callback.

A lookup compares tokens from position zero, orders candidates by longest
match and then recency, and asks each candidate Session to clone the matched
prefix. Each snapshot is tried only at its longest matching boundary; a
retention or resource refusal moves to the next snapshot without retrying
shorter boundaries within the refused snapshot.
The result owns a new independently writable Session and a `matched_tokens`
count. On a miss, open a fresh Session. Submit only the unmatched suffix.
The final requested token is always left for a forward, including on an exact
prompt match, to produce fresh logits under the new sampling policy.

**MLX cold/warm numerical parity is not guaranteed.** Reuse changes prefill
shapes and can change logits, independently of clone-state correctness. Keep
MLX reuse opt-in and restrict experiments to greedy generation until non-greedy
sampling is validated. Greedy token parity is not guaranteed either.

Create one prompt capture per generation. Its wrapper requests a clone at the
complete prompt boundary before forwarding the first nonempty output to the
user callback. Collect the snapshot on the caller thread after generation:

```cpp
auto capture = prefixes.capture_prompt(session, prompt);
auto generation = session.generate_async(
    suffix, config, capture.wrap([&](const GenerationUpdate& update) {
      consume(update); // must not block the engine thread
    }));
generation.wait();
capture.collect();
```

`capture_prompt()` copies the complete token history, including any reused
prefix; its length must fit `Position`. The first callback can expose a pending
prediction or several speculative tokens, so the capture uses this known
prompt boundary rather than the callback token count. Capture binds to the
source's session identity, so both the source Session and capture handle can
move. The helper does not own the source: retain its current owner until
generation completes; destroying that owner still requests closure and
cancels generation. The capture handle can be discarded after wrapping because
the callback shares its state. Keep the PrefixCache alive through `collect()`.

Call `wrap()` once for one generation, and call `collect()` only from the caller
thread after `generation.wait()`. Collection may wait for the queued clone and
inserts the resulting idle snapshot. It returns false when no snapshot was
captured, cloning or insertion was refused, or the handle was already collected.
Disabled caching and capture allocation refusal still forward user callbacks.
Capture attempts cloning at most once and does not suppress user callback
exceptions: those retain the Runner's normal generation-failure behavior.
A failed or cancelled partial prefill produces no snapshot; later cancellation
or a user callback exception does not invalidate an already captured prompt.

`insert()` checks the snapshot position, takes ownership, and evicts the least
recently used entry only after successful insertion. Duplicate keys refresh
recency and release the redundant snapshot. Closing a source does not
invalidate its snapshot; eviction does not invalidate a returned clone.
Snapshots must remain idle after cloning. The caller is responsible for
supplying their exact token history under the same immutable model, method,
KV configuration, and positional semantics. Token-only keys are insufficient
for changing adapters, external embeddings, or image inputs.

Generation/engine metrics count the submitted suffix and actual model work.
The application reports the complete prompt length and matched-token count;
clone lookup latency occurs before `generate_async()` and is outside its
reported generation latency.

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

The generic Session and prefix manager are shared by MLX and future CUDA/Vulkan
executors. A backend using ModuleExecutor supplies a registered cache builder,
registry binding, `BatchControl`, and attention/storage compatible with its
clone and rewind contracts. ModuleExecutor currently passes host token and
position tensors and samples host-visible logits; device boundary copies or
device sampling require appropriate execution integration.

MLX implements the off-graph cache used here. The existing `examples/llm_server`
CUDA workers use another session/execution path and do not automatically gain
this prefix cache. This change does not migrate those workers.

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
