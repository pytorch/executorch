# Private owner rows and shared Ring staging (ABI 3)

CUDA solo, text-only batching exports opt into this layout with
`--private-kv-rows`. It requires the batched-sequence layout and BF16 KV storage;
it does not support DFlash, non-causal attention, or ABI 2's runtime-pitch flag.
Absent this option, existing ABI 1/2 exports and runtime paths are unchanged.
`get_offgraph_kv_sequence_info_abi = 3` selects the matching cache factory.

The export still declares its maximum context, owner count, and step width.
These are addressing and admission ceilings, not initial allocations. Each
layer packs independently sized owner spans `(base_row, capacity)` followed by
one scratch row. A newly opened owner has no rows until it is touched. Its
capacity then grows geometrically from `initial_capacity`, capped by the context
for a flat layer or the sliding window for a Ring layer. A long owner does not
increase other owners' row capacities. Removed owners' spans can be retained
for burst reuse; an explicit idle compaction releases unused spans.

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
ceiling. Logical visibility and reduction order follow the legacy virtual
key sweep; private placement does not reorder keys or change split-K dispatch.

Preparation uploads descriptor contents before execution. Descriptor addresses
stay fixed; stage or history relocation fences the previous step and invalidates
affected bindings and CUDA graphs. Stable allocations replay graphs with new
metadata. The cache verbs and execution must remain serialized by the engine
thread. An aborted ABI 3 forward poisons the cache because the Ring write may
have happened; only a successful fenced `clear()` permits reuse.

## Clone, rewind, and memory accounting

After a successful forward, the runtime archives the actual staged K/V rows
to pinned host transfer buffers and tagged, copy-on-write host pages. This
preserves the legacy rewind and clone window while device Ring history holds
only its window. Commit includes this transfer and synchronization; it is a
performance cost that same-work benchmarks must include. Clone does not mutate
its source. Restore validates the retained window before copying and fences
host page use before reuse or destruction.

`allocated_bytes` counts device K/V history only. `device_staging_bytes` counts
shared device K/V staging separately. `host_undo_bytes` counts unique undo-page
payloads, tags, and pinned archive buffers; it excludes container overhead.
Driver/allocator peaks also include metadata, model weights, activations,
temporary replacement allocations, and retained allocator memory.

`check_private_kv_rows.py` compares identical mini model weights against ABI 2
for exact logits, clone/rewind boundaries, heterogeneous owner lengths, graph
recapture/replay, driver memory contraction, and c1 through c64 same-work timing.
It alternates ABI order across three timing rounds and fails qualification
if their median throughput regresses beyond 2%; a numerical pass alone does
not qualify performance or full-model serving.
