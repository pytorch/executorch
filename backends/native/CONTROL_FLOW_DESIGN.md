# Control Flow in Portable Backend v2

> **Status:** Draft for review.
>
> **Companion to:** [`runtime/PORTABLE_BACKEND_API_PROPOSAL.md`](../runtime/PORTABLE_BACKEND_API_PROPOSAL.md). This document extends the v8.2 architecture to support data-dependent control flow (`torch.cond`, `torch.ops.higher_order.map_impl`, `torch.ops.higher_order.scan`) emitted by ExecuTorch's AOT pipeline.
>
> **Scope:** Runtime-only. AOT emit is unchanged: portable v2 ingests the same `executorch_flatbuffer::Program` that ET's standard runtime does, including its `JumpFalseCall` / `MoveCall` / `FreeCall` instructions.
>
> **Non-goals:**
>   - `torch.while_loop` (upstream emitter raises `InternalError`; revisit when supported).
>   - `DelegateCall` nested *inside* the v2 chain (we accept programs with no nested delegates; partitioner gate enforces this).
>   - GPU-side branching (CUDA dynamic parallelism, Metal indirect command buffers). All control decisions are taken on host.
>   - Memory-pool sharing across mutually-exclusive branches. Both branches' intermediates remain concurrently allocated. Matches ET runtime today; orthogonal to this design.

## 1. Goals

1. **Correctness parity with ET's `Method::execute`** for `cond`, `map_impl`, `scan` over arbitrary segment placements, including cross-runtime predicates and cross-runtime branch bodies.
2. **Async-safe** without regressing the v2 architecture: any work whose result the executor does *not* observe at the jump may continue running in flight.
3. **Precise dependency tracking from day one.** A `JumpFalseStep` waits on exactly the signals that produce its predicate's transitive dependency closure — never a blanket drain.
4. **Backing-agnostic.** The IR adapter (`Graph`) decodes ET's encoded form into a typed instruction stream so the router and executor never touch `executorch_flatbuffer` types. A future schema change replaces the adapter, not the backend.
5. **Ossification-resistant.** Control-flow handling is structured so that an eventual move to region-aware routing (Strategy 3 in the architecture review) reuses every component except the router itself.

## 2. The two invariants

The design rests on exactly two invariants. Everything else falls out.

**Invariant CF-1 — Dispatch and counter state live on host.**
The PC is a host-side variable in the executor. The `JumpFalseStep` runs on host. Every value consumed by a `JumpFalseStep` (predicate) or by ET's emitted loop counters (`executorch_prim::add.Scalar`, `eq.Scalar`, `sub.Scalar`, `et_copy_index.tensor`) materializes on host before use. The router pins the prim ops to the CPU runtime automatically (no other Runtime claims them).

**Invariant CF-2 — Segments do not span jumps.**
A `CompiledSegment` is a contiguous run of `KernelCall` instructions on a single runtime *between* jumps. The router terminates the current segment whenever the next instruction is a `JumpFalseCall` (or `MoveCall` / `FreeCall`). The unconditional-jump pattern ET emits at the end of the true branch (`JumpFalseCall(Bool(False), ...)`) is also a segment break. As a corollary, transfers and mirror values never bridge a branch.

CF-1 makes the executor model identical to a single-threaded interpreter with PC. CF-2 makes the routing problem stay shape-compatible with today's flat `vector<Step>`.

## 3. Architecture additions

```
┌─────────────────────────────────────────────────────────────────┐
│  Graph (api/GraphTypes.h)                                       │
│   + InstructionKind enum                                        │
│   + JumpFalseInfo accessor                                      │
│   + MoveInfo accessor (FreeCall / DelegateCall: rejected)       │
│   + iterate_instructions() yielding (kind, op_idx) pairs        │
└─────────────────────────────────────────────────────────────────┘
                                │ consumed by
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step variant (api/Step.h)                                      │
│   + JumpFalseStep { pred_value_id, dst_step_idx,                │
│                     wait_for, /* no signal */ }                 │
└─────────────────────────────────────────────────────────────────┘
                                │ produced by
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  GreedyRouter additions                                         │
│   + segmenter respects InstructionKind                          │
│   + PredicateLocator: ensure pred is host-resident at jump      │
│   + DepClosure: backward walk through value_producer_seg        │
│   + PCResolver: source PC → step index, second pass             │
└─────────────────────────────────────────────────────────────────┘
                                │ executed by
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PortableBackend_v2::execute                                    │
│   + PC walker with kMaxHops guard                               │
│   + parse_cond_value (port from ET)                             │
│   + observed_terminal_events tracker                            │
└─────────────────────────────────────────────────────────────────┘
```

The five new abstractions in dependency order — each gets one section below:

- **`InstructionKind` + `Graph::iterate_instructions`** (§4) — the IR adapter stops pretending every instruction is a `KernelCall`.
- **`JumpFalseStep`** (§5) — what the router emits and what the executor consumes.
- **`PredicateLocator`** (§6) — a small helper that guarantees the predicate value lives on host at the JumpFalse.
- **`DepClosure`** (§7) — precise per-jump `wait_for` computation.
- **`PCResolver`** (§8) — second-pass mapping from source-PC jump destinations to step-index destinations.

The executor changes (§9) are mechanical once these exist.

## 4. `InstructionKind` and `iterate_instructions`

### 4.1 Problem

`Graph::get_op` currently does an unconditional `static_cast<const KernelCall*>(instr->instr_args())` (`api/GraphTypes.h:585`). Any non-`KernelCall` instruction silently corrupts dispatch. The mem-obj-id precompute in the `Graph` constructor walks chains via `get_op` (`api/GraphTypes.h:381`) and exhibits the same bug. Both must become kind-aware.

### 4.2 New API surface in `api/GraphTypes.h`

```cpp
enum class InstructionKind : uint8_t {
  Kernel,        // KernelCall — only kind a CompiledSegment may contain.
  JumpFalse,     // JumpFalseCall — segment break + control-flow boundary.
  Move,          // MoveCall — segment break; dst = src EValue copy.
  Free,          // FreeCall — segment break; informational (v2 ignores).
  Delegate,      // DelegateCall — REJECTED at init; no nested delegates.
};

struct JumpFalseInfo {
  uint32_t cond_value_id;          // EValue index
  uint32_t destination_pc;         // Source-PC of the jump target
};

struct MoveInfo {
  uint32_t src_value_id;
  uint32_t dst_value_id;
};

class Graph {
 public:
  // Per-instruction (kind, op_idx) iteration. Replaces num_instructions /
  // get_instruction callers that need to handle non-Kernel instructions.
  // O(1) per call; no allocation.
  InstructionKind instruction_kind(size_t chain_idx, size_t op_idx) const;

  // Typed accessors. ET_CHECK if kind doesn't match.
  OperatorCall   get_kernel_call(size_t chain_idx, size_t op_idx) const;
  JumpFalseInfo  get_jump_false (size_t chain_idx, size_t op_idx) const;
  MoveInfo       get_move       (size_t chain_idx, size_t op_idx) const;

  // Convenience: main-chain shortcuts (preserve existing API for callers
  // that already know they're walking only Kernels).
  InstructionKind instruction_kind(size_t op_idx) const {
    return instruction_kind(main_chain_idx(), op_idx);
  }
  // get_instruction(idx) is RETIRED. Callers must select by kind.
};
```

### 4.3 Migration of existing callers

Three call sites today assume Kernel:

1. **`Graph::Graph` mem-obj-id precompute** (`api/GraphTypes.h:381`) — gate on `instruction_kind() == Kernel` before invoking `get_op`.
2. **`InstanceUtils.h:157`** — same guard.
3. **`CpuEngine::compile_segment` and `MetalEngine::compile_segment`** (`cpu/CpuEngine.cpp:515`, `metal/MetalEngine.mm:492`) — these only ever receive `instruction_indices()` from a `CompiledSegment`. By Invariant CF-2 every entry is a Kernel. Add an `ET_DCHECK(instruction_kind == Kernel)` and continue using `get_kernel_call`.

The existing `OperatorCall` API is untouched. `OperatorCall` already documents single-output assumption; out of scope to change.

### 4.4 Why a separate accessor per kind, not a `std::variant`

The hot path in `CpuEngine::compile_segment` runs once per Kernel. A variant access would either branch in the inner loop or force every caller to `std::visit`. Per-kind accessors compile to the same code as today's `static_cast` plus one branch on `instruction_kind` (which the segmenter can hoist out of the kernel-emitting loop because every entry is known-Kernel).

## 5. `JumpFalseStep`

### 5.1 Definition (`api/Step.h`)

```cpp
struct JumpFalseStep {
  // Value to inspect. Always lives on the host runtime by the time this
  // step executes (PredicateLocator guarantees this at routing time).
  uint32_t pred_value_id;

  // Resolved destination as a step index into Plan::steps. Computed by
  // PCResolver in a second pass. The router emits this with a sentinel
  // (kUnresolvedStep) and the resolver patches it.
  size_t dst_step_idx;

  // Precise transitive dependency closure of pred_value_id, expressed
  // as the signals that produce those dependencies.
  std::vector<EventId> wait_for;

  // No signal. JumpFalseStep produces no value; consumers depending on
  // a value are sequenced by that value's producing step's signal, not
  // by the jump itself.
};

using Step = std::variant<ComputeStep, TransferStep, JumpFalseStep>;

inline constexpr size_t kUnresolvedStep = static_cast<size_t>(-1);
```

### 5.2 What JumpFalseStep does NOT contain

- **No `signal`.** A jump doesn't produce a value. Steps that *follow* a jump but depend on values the jump's predicate also depended on already wait on those values' producing steps; the jump is transparent to data-flow.
- **No `runtime_idx`.** Always host. Asserted at execute time.
- **No "true" / "false" branch references.** The jump is one-sided ("if predicate is false, go to dst"). Both branches are reached by linear PC walk, exactly as ET emits them.
- **No mirror-value plumbing.** PredicateLocator resolves locality before this step is constructed.

### 5.3 Schema-level invariant

`PCResolver` MUST resolve every JumpFalseStep before the Plan leaves the router. An unresolved `dst_step_idx` is a router bug, not a runtime condition; assert in debug builds.

## 6. `PredicateLocator`

### 6.1 Responsibility

Guarantee: **for every `JumpFalseCall` in the chain, the predicate value is host-resident at the moment the JumpFalseStep executes.**

### 6.2 Three cases

For each `JumpFalseCall` at source PC `i` with predicate `vid`:

1. **Producer is host or `vid` has no in-delegate producer (graph input / constant / mutable-buffer placeholder).**
   The host pool already homes `vid`. No transfer needed. The JumpFalseStep reads `d->values[vid]` directly; `bind_io` has already made `d->values[vid].toTensor().mutable_data_ptr()` valid.

2. **Producer is on a non-host runtime AND host already has a mirror of `vid`.**
   The router's existing post-Compute mirror machinery (§7.b in the proposal, "producer-side mirror for any value whose home is host but whose producer is non-host") emits a `TransferStep(vid_on_producer → vid_on_host)` after the producing segment. The JumpFalseStep reads the host-side mirror. No new transfer needed. The `wait_for` (computed in §7) will include that transfer's signal.

3. **Producer is on a non-host runtime AND no existing host mirror.**
   PredicateLocator mints a host mirror: `MirrorValueDesc { mirror_id = new, source_value_id = vid_on_producer }`, allocates a host buffer of the predicate's dtype/shape, and emits a fresh `TransferStep` immediately after the producing segment (and necessarily before the JumpFalseStep, since CF-2 places the segment break at the jump). The JumpFalseStep's `pred_value_id` is rewritten to the mirror id.

### 6.3 Where it sits in the router

PredicateLocator runs **after** segment formation and home-provider assignment (so the existing `value_home_provider` and mirror tables are populated) and **before** DepClosure (so DepClosure walks against final mirror identities).

```cpp
class PredicateLocator {
 public:
  PredicateLocator(const Graph& graph,
                   GreedyRouter::SegmentTable& segments,
                   GreedyRouter::MirrorTable& mirrors,
                   GreedyRouter::ValueHomeMap& home);

  // Returns the host-resident value_id to read at jump time. May mutate
  // `mirrors` and append TransferSteps via the caller's StepBuilder.
  uint32_t locate(size_t chain_idx, size_t op_idx,
                  uint32_t raw_pred_vid,
                  StepBuilder& sb);
};
```

The signatures use the router's existing internal types (we don't expose them publicly). The contract is the same as the existing producer-side-mirror code path; the only new thing PredicateLocator does is *demand* a host mirror exists for predicate values, whether or not anything else needed one.

### 6.4 Why this isn't just "always download to host"

For predicates produced on UMA backends (Apple Silicon Metal), the host mirror buffer is an alias of the device buffer (zero-copy). The `TransferStep` becomes a no-op `upload_from_host` whose source pointer equals destination pointer. PredicateLocator doesn't need a special UMA case — the existing `bind_io` / `upload_from_host` path collapses correctly.

For discrete GPUs, the transfer is real (`vkCmdCopyBuffer` or equivalent). One real download per branch is unavoidable; the alternative is GPU-side branching, which is out of scope.

## 7. `DepClosure` — precise `wait_for` computation

### 7.1 The data structures we already have

`GreedyRouter` builds `value_producer_seg: value_id → segment_idx` while forming segments (`routers/GreedyRouter.cpp:131`). It also assigns a `signal: EventId` to each `ComputeStep` and `TransferStep` it emits. These two together let us compute the precise transitive set of steps that must complete before a value is observable on host.

### 7.2 Algorithm

```
DepClosure(pred_vid):
  visited_values: set<uint32_t>
  signals:        set<EventId>
  worklist:       queue<uint32_t> = { pred_vid }

  while worklist not empty:
    v = worklist.pop()
    if v in visited_values: continue
    visited_values.insert(v)

    seg_idx = value_producer_seg[v]
    if seg_idx is none:
      continue  # graph input / constant / mutable buffer — no signal

    for each step that ROUTING associated with seg_idx
        (the segment's ComputeStep + any TransferSteps the router
         emitted for cross-runtime mirrors of values that segment
         consumes / produces):
      if step.signal != kNoEvent:
        signals.insert(step.signal)

    # Recurse: the producing segment depends on its own inputs.
    for input_vid in graph.inputs_of_segment(seg_idx):
      worklist.push(input_vid)

  return signals
```

### 7.3 Complexity and bounding

Worst case is `O(|values| + |segments|)` per JumpFalseCall, because each value is visited once. In practice predicates' dep cones are tiny (a `sum` and a `gt`), so this is microseconds at routing time, never at execute time.

DepClosure runs **once per JumpFalseCall, at routing time**. Its output is baked into the JumpFalseStep's `wait_for`. The executor's per-jump cost is just `wait_for.size()` `Engine::wait` calls.

### 7.4 Edge case: predicate depends on a value computed inside a loop body

When a JumpFalse is the back-edge of a loop, its predicate depends on an `eq.Scalar` whose input is `add.Scalar`'s output. Both prim ops are pinned to host (Invariant CF-1) and produce no signals (sync host ops emit a no-op event signaled before the next step). DepClosure terminates at host-only producers without recursing into device land.

When a JumpFalse predicate depends on a tensor produced by the loop body (e.g., a hypothetical "early exit" pattern), DepClosure correctly walks back through the body's compute step and adds its signal — meaning the executor waits on the body before testing the predicate. This is correct.

### 7.5 Why precise closure beats drain-all

Two reasons:

1. **Async backends benefit linearly.** On a discrete GPU plan with 30 in-flight kernels, a drain-all `JumpFalseStep` serializes all 30; a precise closure may need to wait on only 2.
2. **It's not actually more code.** `value_producer_seg` already exists. The closure walk is ~40 LOC. The "savings" of drain-all (~80 LOC) is illusory because we'd still need to track which signals are outstanding, which is the same machinery.

## 8. `PCResolver` — source-PC → step-index resolution

### 8.1 Problem

ET emits `JumpFalseCall.destination_instruction = <source PC>`. Our `Plan::steps` is denser than the source instruction stream (multiple TransferSteps may be inserted, FreeCalls dropped, etc.) and sparser in other places (a segment of N kernels collapses to one ComputeStep). PCResolver maps every source PC to "the step index at-or-after that PC, in execution order."

### 8.2 Algorithm (single second pass)

```
PCResolver(plan, segments):
  pc_to_step: array<size_t> of size num_instructions, initialized to kUnresolved

  # Walk steps in plan order; each step records its lowest source PC.
  for step_idx, step in enumerate(plan.steps):
    pc = source_pc_of(step)  # ComputeStep: segment.first_pc;
                             # TransferStep: pc of the JumpFalse / segment-edge
                             # that triggered it; JumpFalseStep: its own pc.
    pc_to_step[pc] = min(pc_to_step[pc], step_idx)

  # Backward fill: every PC inherits the next-occupied PC's step.
  next_step = plan.steps.size()  # sentinel: past-the-end
  for pc in reverse(0 .. num_instructions):
    if pc_to_step[pc] != kUnresolved:
      next_step = pc_to_step[pc]
    else:
      pc_to_step[pc] = next_step

  # Resolve every JumpFalseStep.
  for step in plan.steps:
    if isinstance(step, JumpFalseStep):
      step.dst_step_idx = pc_to_step[step.unresolved_dst_pc]
      assert step.dst_step_idx != kUnresolvedStep
```

### 8.3 What "source_pc_of(step)" means

- `ComputeStep`: the lowest source PC in `segment->instruction_indices()`.
- `TransferStep`: the source PC of the producing/consuming `KernelCall` it serves (the producer for post-Compute downloads, the consumer for pre-Compute uploads).
- `JumpFalseStep`: the source PC of the originating `JumpFalseCall`.

The router records `source_pc_of` on each step at construction time (one extra `uint32_t` per step) so PCResolver doesn't need to recompute it.

### 8.4 Why backward fill is correct

ET's emitter only ever generates jumps to PCs that are themselves the start of some instruction (segment boundaries or branch joins). After CF-2 (segments don't span jumps), any PC at-or-after a target that *is* the start of a step has its index recorded directly. PCs that aren't step starts are interior to a segment (impossible to be a jump target by CF-2) or skipped instructions like `FreeCall` (forwarded by backward fill to the next real step). The fill is exact, not approximate.

### 8.5 Why this is a separate component

PC resolution is the one place where the source instruction stream and the planned step stream must reconcile. Hiding it in the segmenter's main loop would couple "where do I break a segment?" with "where does this jump land?" — two questions that have nothing to do with each other. Keeping `PCResolver` as a single-pass post-processor keeps the segmenter pure and makes PC resolution unit-testable in isolation against synthetic Plans.

## 9. Executor changes

### 9.1 PC walker (`PortableBackend_v2.cpp::execute`)

```cpp
constexpr size_t kMaxHops = 10'000'000;  // ET parity (method.cpp:37)

size_t pc = 0;
size_t hops = 0;
Error first_err = Error::Ok;

while (pc < d->plan.steps.size()) {
  if (++hops > kMaxHops) {
    first_err = Error::InvalidProgram;
    break;
  }

  const Step& s = d->plan.steps[pc];
  size_t next_pc = pc + 1;

  if (auto* jf = std::get_if<JumpFalseStep>(&s)) {
    // Drain precise dependency closure.
    for (EventId id : jf->wait_for) {
      if (id >= d->plan.events.size()) continue;
      auto& slot = d->plan.events[id];
      if (slot.event && slot.owner) {
        Error e = slot.owner->wait(slot.event.get());
        if (e != Error::Ok && first_err == Error::Ok) first_err = e;
      }
    }
    if (first_err != Error::Ok) break;

    // Read predicate. By PredicateLocator's contract, pred_value_id
    // is always host-resident here.
    Result<bool> r = parse_cond_value(d->values[jf->pred_value_id]);
    if (!r.ok()) { first_err = r.error(); break; }

    // Track for observed_terminal_events (see §9.3).
    d->observed_jumps.push_back(pc);

    next_pc = r.get() ? pc + 1 : jf->dst_step_idx;
  } else {
    Error e = execute_step(d, s);
    if (e != Error::Ok) { first_err = e; break; }
    // Track for observed_terminal_events.
    if (auto* cs = std::get_if<ComputeStep>(&s)) {
      if (cs->signal != kNoEvent) d->observed_signals.push_back(cs->signal);
    } else if (auto* ts = std::get_if<TransferStep>(&s)) {
      if (ts->signal != kNoEvent) d->observed_signals.push_back(ts->signal);
    }
  }

  pc = next_pc;
}
```

### 9.2 `parse_cond_value` (anonymous-namespace port from ET)

Direct lift of `runtime/executor/method.cpp:253`. ~35 LOC. Accepts `Tensor[Bool]` (returns `false` if any element false) and scalar `Bool` (passthrough). Returns `Error::InvalidProgram` for any other EValue kind.

We port (not link) for two reasons: (a) `parse_cond_value` is a `static` helper inside ET's `method.cpp` and not exposed; (b) keeping the implementation local lets v2 evolve its predicate semantics independently if needed (e.g., supporting `Tensor[uint8]` later).

### 9.3 Terminal events under branches

Today `Plan::terminal_events` is a static set ("the last signal each Engine emitted, statically"). With branches, some terminal signals are never produced (the skipped branch's). Two options:

- **Option A — observed-set:** the executor maintains `observed_signals` (signals it actually issued this run) and `observed_jumps` (which JumpFalseSteps actually fired). At end of `execute`, `wait` only on signals in `observed_signals` that are also in `plan.terminal_events`. Cost: one `vector<EventId>::push_back` per step. Memory: bounded by `plan.steps.size()`.

- **Option B — wait-if-set:** add a "signaled?" bit to each `EventSlot`. Producers set it before signaling; the executor `wait`s only on slots whose bit is set this execute, then clears. Costs less per step (just a bit set) but requires backend cooperation to set the bit.

**Choose A.** Option B leaks a v2-internal concern into every Engine implementation. Option A is purely executor-local and the per-step push_back is negligible compared to kernel cost.

Reset both `observed_signals` and `observed_jumps` at the top of every `execute`.

### 9.4 Failure / poisoning

Existing v2 contract: any step error sets `d->poisoned = true` and returns. Unchanged. A failure inside `execute_step` for a step that won't be visited again in this execute (because a JumpFalse later skipped it) cannot occur — failures happen before the JumpFalse, so the executor exits without reaching the jump.

## 10. Router-level pseudocode (full picture)

```
GreedyRouter::route(graph, providers, instances, ndm, options) -> Plan:

  # --- Phase 1: per-Kernel routing decision (existing) ---
  for chain_idx, op_idx in graph.iterate_instructions():
    kind = graph.instruction_kind(chain_idx, op_idx)
    if kind == InstructionKind::Kernel:
      assign provider for graph.get_kernel_call(...)

  # --- Phase 2: segment formation (modified — CF-2) ---
  segments = []
  current_seg = None
  for chain_idx, op_idx in graph.iterate_instructions():
    kind = graph.instruction_kind(chain_idx, op_idx)
    if kind != InstructionKind::Kernel:
      flush(current_seg); current_seg = None
      record_control_instruction(chain_idx, op_idx, kind)
      continue
    p = chosen_provider[op_idx]
    if current_seg is None or current_seg.provider != p:
      flush(current_seg)
      current_seg = new_segment(p, first_pc=op_idx)
    current_seg.add_kernel(op_idx)
  flush(current_seg)

  # --- Phase 3: home-provider + mirror tables (existing) ---
  build value_producer_seg, value_home_provider, mirror_values

  # --- Phase 4: emit ComputeSteps + intra-graph TransferSteps (existing) ---
  for seg in segments:
    emit pre-Compute uploads for cross-runtime inputs
    emit ComputeStep for seg
    emit post-Compute downloads for cross-runtime outputs

  # --- Phase 5 (NEW): PredicateLocator + JumpFalseStep emission ---
  for (chain_idx, op_idx, kind) in control_instructions:
    if kind == InstructionKind::JumpFalse:
      info = graph.get_jump_false(chain_idx, op_idx)
      host_pred_vid = PredicateLocator.locate(chain_idx, op_idx,
                                              info.cond_value_id, sb)
      jf = JumpFalseStep {
        pred_value_id     = host_pred_vid,
        dst_step_idx      = kUnresolvedStep,
        wait_for          = {},  # filled by Phase 6
        unresolved_dst_pc = info.destination_pc,  # for Phase 7
        source_pc         = op_idx,
      }
      emit jf
    elif kind == InstructionKind::Move:
      info = graph.get_move(chain_idx, op_idx)
      emit TransferStep { src_value_id=info.src, dst_value_id=info.dst,
                          src_idx=host, dst_idx=host, source_pc=op_idx }
    elif kind == InstructionKind::Free:
      pass  # v2 ignores; memory plan handles deallocation
    elif kind == InstructionKind::Delegate:
      return Error::NotSupported  # partitioner contract violated

  # --- Phase 6 (NEW): DepClosure ---
  for jf in plan.steps where isinstance(JumpFalseStep):
    jf.wait_for = DepClosure(jf.pred_value_id)

  # --- Phase 7 (NEW): PCResolver ---
  PCResolver(plan, segments)  # patches every JumpFalseStep.dst_step_idx

  # --- Phase 8: alloc_plans + constants + IO bindings (existing) ---
  ...

  # --- Phase 9: terminal_events (modified) ---
  # Plan.terminal_events is now "the *candidate* terminal signal set"
  # — the union of last-signal-on-each-Engine across both branches of
  # every cond. The executor's observed_signals filter narrows this to
  # the actually-issued subset at execute time.
  build plan.terminal_events as: for each Engine, the union of
    last-signals across all steps targeting it (regardless of branch).
```

## 11. Schema and partitioner contract

### 11.1 Partitioner gate

The portable v3 partitioner (`partitioner/portable_partitioner.py`) is the gate. It must:

1. **Accept** subgraphs containing `torch.ops.higher_order.cond / map_impl / scan` whose branches are otherwise v2-supported.
2. **Reject** subgraphs containing:
   - `torch.ops.higher_order.while_loop` (until upstream emit support exists)
   - Nested `executorch_call_delegate` (no DelegateCall inside our chain)
   - Higher-order ops we can't lower (any non-Kernel ET emits other than JumpFalseCall / MoveCall / FreeCall)

Rejection at the partitioner level keeps the runtime contract minimal: v2 may assume its delegate's chain contains only `KernelCall | JumpFalseCall | MoveCall | FreeCall`.

### 11.2 AOT verification

Add a one-pass post-emit verifier (Python, in `preprocess_v3.py`) that walks the emitted chain and asserts the runtime contract. Catches partitioner bugs at AOT time rather than at backend init.

## 12. Testing strategy

### 12.1 Unit tests (sub-component, no kernels)

- **`PCResolver`**: synthetic Plan with hand-written ComputeStep/TransferStep/JumpFalseStep mix; verify `dst_step_idx` resolution including:
  - Forward jumps (cond)
  - Backward jumps (map/scan loop back-edge)
  - Jump target lands on a TransferStep
  - Jump target lands on a FreeCall PC (must skip to next real step)
- **`DepClosure`**: synthetic graph with hand-built `value_producer_seg`; verify closures of:
  - Predicate is a graph input → empty closure
  - Predicate is a constant → empty closure
  - Predicate produced by a single host segment → 0 or 1 signals (host signals may be no-op)
  - Predicate produced by Metal segment, depending on host segment, depending on Metal segment → 2 signals
- **`PredicateLocator`**: predicate on host (no-op), predicate on Metal with existing host mirror (reuse), predicate on Metal with no mirror (mint).

### 12.2 Integration tests (full execute)

Mirror ET's `exir/tests/control_flow_models.py`:

- `FTCondBasic` on CPU-only, then on Metal (force predicate to Metal via routing knob).
- `FTMapBasic` on CPU-only, then with body kernels routed to Metal.
- `FTScanBasic` on CPU-only.
- Nested cond inside cond.
- Cond inside map body.
- Maliciously-formed JumpFalse with self-target (must hit `kMaxHops` and return `InvalidProgram`).

Compare bit-identical outputs against ET's `Method::execute` for the same `.pte`.

### 12.3 Async correctness tests

- A program where a JumpFalse predicate depends on segment A but a long-running segment B is in-flight. Assert via instrumentation that the JumpFalse `wait`s only on A's signal, never on B's.
- A loop body that issues async work; assert no signal-slot exhaustion across iterations.

## 13. Forward-compatibility hooks

The design preserves the option to evolve into Strategy 3 (region-aware routing) without rewriting any of the components introduced here:

- **`InstructionKind` and per-kind accessors** survive unchanged in a region-aware world; regions are built *on top of* the typed instruction stream.
- **`JumpFalseStep` becomes a leaf-region terminator** rather than a top-level Step; same fields, same executor logic.
- **`PredicateLocator` / `DepClosure` / `PCResolver`** all operate within a single region's flat instruction range. A region-aware router invokes them per-region.
- **The executor's PC walker** becomes a per-region walker; the outer driver pushes/pops region frames. The inner loop is byte-identical.

So this design is the strict subset of Strategy 3 that fits today's flat `vector<Step>`. None of it needs to be unwound.

## 14. Acceptance criteria

1. All existing v2 tests pass unchanged.
2. New integration tests (§12.2) pass with bit-identical outputs vs. ET's `Method::execute`.
3. `kMaxHops` guard fires on a hand-crafted self-jumping program (returns `InvalidProgram`).
4. `[mem]` debug logs trace JumpFalseStep predicate locality (host vs. mirror) and `wait_for` size, enabling profiling of branch overhead.
5. The partitioner verifier (§11.2) rejects programs with `while_loop`, `DelegateCall` in chain, or other unsupported instructions before they reach the runtime.

## 15. Component LOC estimate

| Component | File | New LOC |
|---|---|---|
| `InstructionKind`, kind accessors | `api/GraphTypes.h` | ~110 |
| `JumpFalseStep`, variant extension | `api/Step.h` | ~20 |
| Plan field additions (`source_pc` per step) | `api/Plan.h` | ~10 |
| Segmenter CF-2 enforcement | `routers/GreedyRouter.cpp` | ~70 |
| `PredicateLocator` | `routers/GreedyRouter.cpp` | ~120 |
| `DepClosure` | `routers/GreedyRouter.cpp` | ~50 |
| `PCResolver` | `routers/GreedyRouter.cpp` | ~60 |
| Phase 5 control-instruction emission | `routers/GreedyRouter.cpp` | ~50 |
| Executor PC walker + observed-set | `PortableBackend_v2.cpp` | ~80 |
| `parse_cond_value` port | `PortableBackend_v2.cpp` | ~35 |
| AOT verifier | `preprocess_v3.py` | ~60 |
| Unit + integration tests | `test_*.cpp` | ~500 |
| **Total** | | **~1165** |

## 16. Open questions

1. **`MoveCall` semantics.** ET emits `MoveCall` to copy one EValue's contents into another (used by some HOP output plumbing). Treating it as `TransferStep(host, host)` is correct but conflates "memory move" with "cross-runtime transfer." Acceptable now; revisit if profiling shows the host-host TransferStep path is too heavy for what is essentially a pointer copy. Alternative: introduce a `MoveStep` variant. Adds 30 LOC; defer until needed.

2. **Constant-fold trivial JumpFalseSteps?** ET's unconditional-jump trick uses a constant `Bool(False)` predicate. The router could detect this and replace the JumpFalseStep with a router-time PC rewrite (no runtime check at all). Saves one `parse_cond_value` per cond. Tiny win; consider in a follow-up.

3. **Memory-pool sharing across cond branches.** Out of scope per §1, but worth flagging as the highest-value follow-up. Requires the upstream memory planner to model branch alternatives, which is a much larger change.

4. **Profiling integration.** Should JumpFalseStep show up in the v2 trace as a distinct event, or be folded into the surrounding compute? Recommend distinct: makes branch-vs-body cost attribution possible.