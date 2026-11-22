# Private owner rows and shared Ring staging (ABI 3)

CUDA solo, text-only batching exports opt into this layout with
`--private-kv-rows`. It requires the batched-sequence layout and BF16 KV storage;
it does not support DFlash, non-causal attention, or ABI 2's runtime-pitch flag.
Absent this option, existing ABI 1/2 exports and runtime paths are unchanged.
`get_offgraph_kv_sequence_info_abi = 3` selects the matching cache factory.

The export still declares its maximum context, owner count, and step width.
These are addressing and admission ceilings, not initial allocations. Each
layer stores independently sized owner spans `(base_row, capacity)` in an arena,
plus one scratch row. An owner ID with no retained span has no rows until touched. Its
capacity then grows geometrically from `initial_capacity`, capped by the context
for a flat layer or the sliding window for a Ring layer. A long owner does not
increase other owners' row capacities. Geometric arena slack amortizes owner
admission; unrelated spans keep their positions. Growth uses free regions
disjoint from old payloads, so it can relocate one owner without changing device
addresses or recapturing graphs. Arena exhaustion or fragmentation at the
addressing ceiling creates a packed replacement. Removed owners' spans can be
retained for burst reuse. Reusing an ID can temporarily inherit that capacity;
explicit compaction plans live spans from their current requirements and the
initial capacity, releasing inherited excess as well as unused spans and arena
slack. A wrapped Ring retains its window and modulus during compaction.

The executor's automatic idle heuristic still uses owner counts. Closing a few
long owners while many short owners remain may therefore retain their spans
until explicit compaction. Byte-aware idle reclamation is a separate allocator
maintenance change; this layout does not yet provide that policy.

## Frozen program and runtime contract

All layer K/V constants retain the legacy BSHD virtual shape. The allocation
backing that shape can be smaller: only the private operators may access it,
using the runtime descriptors. No generic tensor copy or linear indexing over
the declared extent is valid. Constant metadata from a different cache layout
is rejected at load time.

`seq_cols[T]` supplies stable owner IDs in the original query order and
`seq_info[2 * max_seqs, 4]` describes query runs. Each layer's
`owner_layout[max_seqs + 1, 2]` contains its private span descriptors.
`owner_step[max_seqs + 1, 3]` contains `(stage_start, absolute_start,
token_count)`. The last owner is padding and never contributes a visible
key. `stage_rows[T]` is a unique permutation that packs each owner's new K/V
rows contiguously without changing query order.

Flat layers write their private span before attention. Ring layers first put
new K/V into shared per-layer staging, attend to frozen private history plus
that staging, and only then write the newest retained rows into the Ring.
This prevents a long chunk from overwriting history needed by its earlier
queries, without reserving `max_write - 1` extra device rows for every owner.
Stage allocations grow with the actual prepared step width, up to the export
ceiling. Attention selects a history or staging address before loading each
row, avoiding separate pipelined K/V tiles and their shared-memory overhead.
Logical visibility and reduction order follow the legacy virtual
key sweep; private placement does not reorder keys or change split-K dispatch.
For D=256, pipeline buffering is reduced on targets below 160 KiB of effective
shared memory. The original pipeline remains on large-SMEM targets; tiles are
unchanged on both.

Preparation uploads descriptor contents before execution. Descriptor addresses
stay fixed; stage or arena address changes fence the previous step and invalidate
bindings and CUDA graphs. Within-arena relocation copies only that owner's
retained rows, preserving every old source until commit. Failed preparation
fences queued copies before returning, including copies into unused arena rows.
Stable allocations replay graphs with new
metadata. The cache verbs and execution must remain serialized by the engine
thread. An aborted ABI 3 forward fences queued work and retires only the owners
in that step, whose Ring writes may have happened. `seq_rm` releases those
owners for reuse; unrelated owners can keep serving. A failed device fence
poisons the whole cache until a successful fenced `clear()`.

## Clone, rewind, and memory accounting

After a successful forward, the runtime archives the actual staged K/V rows
to pinned host transfer buffers and tagged, copy-on-write host pages. This
preserves the legacy rewind and clone window while device Ring history holds
only its window. Commit includes this transfer and synchronization; it is a
performance cost that same-work benchmarks must include. Clone does not mutate
its source. Restore validates the retained window before copying and fences
host page use before reuse or destruction. Rewind restores only the prefix
overwritten since the target position, and a no-op rewind performs no copies.
Clone copies still-resident Ring rows device-to-device and reads only evicted
rows from host history. The unused oldest Ring slot is masked out of both
key and value loads, so an unwritten slot cannot inject NaNs into attention.

`allocated_bytes` counts device K/V history only. `device_staging_bytes` counts
shared device K/V staging separately. `host_undo_bytes` counts unique undo-page
payloads, tags, and pinned archive buffers; it excludes container overhead.
Driver/allocator peaks also include metadata, model weights, activations,
temporary replacement allocations, and retained allocator memory.

Undo pages trade host RAM for device headroom; they are not free or hidden from
the accounting. For each Ring layer, the retained host bound is
`max_seqs * ceil(min(context, W + max_write - 1) / 64) *
(2 * 64 * row_bytes + 64 * sizeof(int32_t)) + 2 * max_write * row_bytes`,
including BF16 K/V pages, row tags, and pinned transfer staging. There is no
separate RAM-budget knob; the session and export limits bound this storage.
For 39 Ring layers with W=2048 and 2 KV heads of dimension 128, max_write=4096
and 64 owners, this is about 14.8 GiB of retained host storage. The checker
asserts each profile's corresponding bound and reports host bytes beside the
device saving, including peak accounted undo storage across the workload.

`check_private_kv_rows.py` compares identical mini model weights against ABI 2
for exact logits, clone/rewind boundaries, heterogeneous owner lengths, graph
recapture/replay, driver memory contraction, and c1 through c64 same-work timing.
Four rounds balance AB/BA order. Timing covers steady decode, staggered
admission/growth, and clone/rewind cycles including GPU completion. A median
throughput regression beyond 2% fails performance qualification. Inputs vary by
owner, position and session lifetime so row-placement and stale-owner errors
change the selected logits; matching non-finite output is rejected. PR CI reports
timing independently from its blocking numerical/memory/graph checks. Automatic
CI reports timing; an explicit workflow dispatch can enforce it with
`enforce_private_kv_performance=true`. The same checker enforces timing when
invoked without `--report-performance`. A green report-only check does not qualify performance
or full-model serving.

`--profile mg-ring` uses 52 layers, 32 query heads, two KV heads of dimension
128, the production `[2048,2048,2048,0]` attention pattern, a 4096-token write
budget, and a 128K logical context. Embedding/MLP dimensions and vocabulary
remain small: this qualifies production attention/cache geometry, not 30B
quantized GEMM or full-model serving performance. It includes 4096-token
writes and restores evicted Ring history at the legal 8192-to-4096 rollback
floor, then measures c1 through c64 with saturated Ring history. Its separate
A100 CI job preserves these limits rather than reducing them to fit the
mini test's A10G. Reproduce it with:

```bash
python examples/models/muse-glimmer/tests/check_private_kv_rows.py \
  --probe /path/to/muse_glimmer_runtime_kv_probe \
  --output-dir /path/to/new-qualification-directory --profile mg-ring
```

Omitting `--report-performance` enforces the same 2% performance gate.

The probe also reports cumulative CPU wall-time counters for preparation,
commit, archive transfer plus its device wait, and host-page append. The checker
differences these counters per operation and reports balanced-round medians.
Archive transfer and append are subcomponents of commit, not additional times
to sum with it. These measurements locate cache-side overhead; they are not
isolated GPU-kernel timings. Clear preserves counters for a complete run.

A separate 4K-context phase keeps 48 short owners alive while closing 16 long
owners. It requires at least 16 MiB less live KV and at least 8 MiB less allocator
reservation and driver-used memory after compaction, exceeding pool page
granularity. Lower live-byte counters without actual driver release do not pass.
