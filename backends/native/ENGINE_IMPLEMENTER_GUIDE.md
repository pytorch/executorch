# Engine Implementer Guide

This document is the contract for implementing a new backend engine in `backends/native`. If you're adding a CUDA engine, a Vulkan engine, an NPU engine, or any other compute runtime, start here.

**Glossary:** "vid" = `value_id`, the `uint32_t` index into the central `EValue` array. Original graph values have vids `< graph.num_values()`; router-minted mirror vids live past that range.

---

## 1. Pick your base class

Two abstract bases:

- **`Engine`** — for the host pool only. Lives in `core/Engine.h`. Has no transfer methods.
- **`DeviceEngine : public Engine`** — for any compute runtime (CPU, Metal, CUDA, Vulkan, NPU). Adds two pure virtuals: `upload_from_host` and `download_to_host`.

If you're writing a compute runtime, **inherit `DeviceEngine`**. Examples: `CpuEngine`, `MetalEngine`.

The host pool is special and there's already exactly one (`HostPoolEngine`). You will not write a second one.

---

## 2. The four `MemoryKind` values

Every `AllocRequest` the router sends you carries a `MemoryKind`. You dispatch on this enum alone — no other field gates behavior.

| Kind | Who handles it | Storage owner | Per-execute work |
|---|---|---|---|
| `HostExtern` | HostPool only | Caller (per-execute) | HostPool re-aliases its wrapper to caller's pointer |
| `HostMirror` | HostPool by default; an engine MAY claim (e.g., CUDA pinned) | Delegate (init or lazy) | None — stable |
| `DeviceMirror` | The targeted device engine | Delegate (init or lazy) | None — stable |
| `DeviceOnly` | The targeted device engine | Delegate (init or lazy) | None — stable |

You will receive `HostExtern` requests **only** if you are HostPool. You won't.

`HostMirror` is biddable in the protocol: an engine MAY claim it (e.g., CUDA could allocate as `cudaMallocHost` for pinning) and HostPool floors. **In the current router, HostMirror requests are emitted only on HostPool's plan**, so device engines won't see them today — the cross-engine claim path is plumbing-ready but not yet exercised. If you opt in for it, the router will need to emit shared plans (a small change).

You will receive `DeviceMirror` and `DeviceOnly` requests for any segment routed to your runtime. You **must** claim these — declining is a router/init bug.

Each request also carries a `mem_obj_id`. **`>= 0` means planned** (group with other vids of the same id, share storage, size from AOT). **`< 0` means unplanned/dynamic** (claim the request, defer allocation to the first `resize_tensor` call). See §3 and §8.

---

## 3. `allocate_buffers` — the init-time storage protocol

Signature:

```cpp
Error allocate_buffers(
    Span<const AllocRequest> requests,
    Span<EValue> values,
    Span<AllocClaim> out_claims) override;
```

You receive a list of requests targeted at you. For each one, write `out_claims[i] = Claimed | Declined` and, if Claimed, allocate a Buffer.

### `mem_obj_id` — two regimes

Each `AllocRequest` carries a `mem_obj_id`:

- **`mem_obj_id >= 0` (planned):** The AOT memory planner assigned this slot. Multiple requests sharing the same id MAY share storage (the planner determined their lifetimes don't overlap). The engine groups by id and sizes one Buffer per group from the per-vid `nbytes` (read from the central TensorImpl).
- **`mem_obj_id < 0` (unplanned/dynamic):** No AOT-known size. The vid is dynamic-shaped or unbounded. **Claim ownership but defer allocation** — record the vid in your internal table with a null Buffer pointer, then allocate lazily on the first `resize_tensor()` call.

**Critical:** If you claim a `DeviceMirror`, `DeviceOnly`, or `HostMirror` request with `mem_obj_id < 0`, **you MUST be able to size it later via `resize_tensor()`**. If your engine doesn't support dynamic resize, you have two clean failure paths:

- **Per-request decline:** return `out_claims[i] = Declined` for the unhandleable request. NativeBackend's `materialize_buffers` validates that `DeviceMirror` / `DeviceOnly` requests were Claimed and fails init with `Error::NotSupported` if any was Declined. Granular and explicit; the diagnostic identifies the offending vid and engine.
- **Whole-batch failure:** return `Error::NotSupported` from `allocate_buffers` itself. Init fails with that error propagated. Heavier hammer; useful when several requests in the batch are unhandleable.

Either way, init fails cleanly rather than crashing at first execute. The router should never route a dynamic vid to an engine that can't handle it; failing at init makes that contract explicit. (Today's router has no `supports_dynamic_shapes()` query, so this is an implicit trust; will become a real capability check when dynamic shapes are exercised.)

`HostExtern` with `mem_obj_id < 0` is unproblematic — caller provides storage at bind time, no engine allocation involved.

### What you must do per request

**For `HostMirror` (only seen if you opted in via the future cross-engine plan):**
- Either Claim and allocate (e.g., `cudaMallocHost`), or Decline.
- If Claimed: write the host pointer into `values[req.value_id].toTensor().unsafeGetTensorImpl()->set_data(ptr)`. This is how cross-engine consumers (like a paired DeviceMirror partner) find your host pointer.

**For `DeviceMirror`:** do these in order:

1. **Materialize the TensorImpl** if `values[req.value_id]` isn't already a Tensor. The router minted `req.value_id` past the end of the original graph value space, so the EValue starts empty. Use the `materialize_mirror_tensor()` helper:
   ```cpp
   if (!values[req.value_id].isTensor()) {
     mirror_tensor_metas_.push_back(materialize_mirror_tensor(
         values, req.value_id, req.host_mirror_value_id));
   }
   ```
   Store the returned `TensorImplStorage` in a member vector that lives as long as the engine. (You can hoist this to a separate first pass over `requests` if it makes your loop cleaner; CpuEngine and MetalEngine both do.)
2. **Record the host->mirror mapping** for `set_io_bindings` to find later: `host_to_mirror_id_[req.host_mirror_value_id] = req.value_id`. Do this **before** any deferral check below — the mapping is needed even for vids whose buffers are deferred.
3. **Mark `out_claims[i] = Claimed`.**
4. **If `mem_obj_id < 0`:** insert `nullptr` into `value_to_buffer_[req.value_id]` and continue. `resize_tensor` will allocate later (and will perform the partner-pointer / UMA-collapse check then). If your engine doesn't override `resize_tensor`, return `Declined` for this request — NativeBackend will fail init with a clear diagnostic. (Or fail the whole allocate_buffers call with `Error::NotSupported`.)
5. **Else (planned):** read `values[req.host_mirror_value_id].toTensor().mutable_data_ptr()` to find the host partner's pointer. If non-null, you may UMA-collapse (alias your device buffer to the host bytes; zero-copy). If null (discrete GPU), allocate fresh device memory.
6. **If your Buffer is host-addressable**, write its host pointer into `values[req.value_id]`.

**For `DeviceOnly`:**
- Always Claim (unless `mem_obj_id < 0` and you can't support `resize_tensor` — in that case return `Declined`; see below).
- `values[req.value_id]` is already a Tensor (it's an original graph vid).
- If `mem_obj_id < 0`: defer to `resize_tensor` (insert null placeholder in `value_to_buffer_`). If your engine doesn't override `resize_tensor`, return `Declined` and NativeBackend will fail init with a descriptive error.
- Else: allocate a Buffer; write `data_ptr` if host-addressable.

### Internal bookkeeping

You own a private `value_to_buffer_: unordered_map<vid, Buffer*>`. Populate it for every Claimed request — with `nullptr` for deferred (mem_obj_id < 0) requests so `resize_tensor` knows the vid is yours. **Nobody outside your engine reads this map.**

If you store DeviceMirror mappings, also populate `host_to_mirror_id_: unordered_map<host_vid, mirror_id>` so `set_io_bindings` can find your mirror for graph IO vids.

### What you must NOT do

- Don't read or write any vid that isn't in your request list.
- Don't try to coordinate with other engines. The protocol is single-allocator-per-kind.
- Don't decline a `DeviceMirror` or `DeviceOnly` request — that's a routing bug, return Error::Internal.

---

## 4. `set_io_bindings` — register your per-execute IO interest

Signature:

```cpp
Error set_io_bindings(
    Span<const InputBinding> graph_inputs,
    Span<const OutputBinding> graph_outputs) override;
```

Called once at init, after `allocate_buffers`. You walk the graph IO lists and decide which IO slots concern you. Store the results internally as `(graph_io_idx, internal_vid)` pairs.

You care about a graph IO vid if:
- You allocated a `DeviceMirror` for it (look up `host_to_mirror_id_`); the internal vid is the mirror_id.
- OR `handles_input_directly(vid)` returns `true` for an input (or `handles_output_directly` for an output); the internal vid is the original vid.

Default implementation is no-op. **Override only if your engine has IO work to do per execute.**

### When to leave the default no-op

If you're a host-resident engine that reads through the central EValue's `data_ptr` at execute time (HostPool sets it during its bind), you don't need any IO bindings. **`CpuEngine` does this** — it overrides `handles_*_directly` to return `true` (so the router doesn't mint a mirror for it) but keeps `set_io_bindings` as the default no-op (no work needed per execute).

---

## 5. `bind_inputs` / `bind_outputs` — per-execute IO

Signatures:

```cpp
Error bind_inputs(Span<EValue> values, Span<EValue* const> input_args) override;
Error bind_outputs(Span<EValue> values, Span<EValue* const> output_args) override;
```

Called every execute, on every engine, with the full caller IO arrays. You self-filter by walking your stored bindings:

```cpp
for (const auto& b : io_input_bindings_) {
  if (b.graph_idx >= input_args.size()) continue;
  if (!input_args[b.graph_idx]) continue;
  // Re-alias your Buffer for b.internal_vid to caller's pointer.
  // Update values[b.internal_vid] data_ptr if your Buffer is host-addressable.
}
```

### The contract

- After your `bind_*` returns, all your IO bindings should reflect the current caller pointers.
- If your buffer is host-addressable and you alias caller's storage, write the caller pointer into `values[internal_vid].data_ptr` so other code sees it.
- If your buffer is discrete VRAM, just remember caller's pointer for the upcoming TransferStep (don't write anything to the central EValue's `data_ptr`).

### What "the engine doesn't bind a vid" means

`bind_inputs` and `bind_outputs` are called on **every** engine every execute. The engine self-filters via its `io_*_bindings_` table; vids not in the table are simply not iterated.

Which vids land in your table is determined at `set_io_bindings` time by your `handles_*_directly` decisions and your claimed `DeviceMirror` requests:

- `handles_input_directly = false` (default) for a graph IO vid → the router emitted a `DeviceMirror` + a `TransferStep` for it. Your `set_io_bindings` records the mirror_id; your `bind_inputs` re-aliases the mirror per execute. The TransferStep handles byte movement.
- `handles_input_directly = true` → the router emitted no mirror and no TransferStep. Your `bind_inputs` is the only chance to wrap caller's pointer in your own Buffer view. Use it.
- The vid isn't yours at all (you don't have a mirror, and you returned `false`) → your `bind_*` does nothing for that vid; some other engine handles it.

---

## 6. `handles_input_directly` / `handles_output_directly`

```cpp
bool handles_input_directly(uint32_t graph_input_value_id) const override;
bool handles_output_directly(uint32_t graph_output_value_id) const override;
```

Per-vid capability query, called at routing time. Default: `false`.

| Return | Meaning | Use case |
|---|---|---|
| `false` (default) | Router mints a `DeviceMirror` + `TransferStep` for this vid. Your `bind_*` is NOT called for it. | Discrete GPU; vids you can't directly consume the caller pointer for. |
| `true` | No mirror, no TransferStep. Your `bind_*` IS called for this vid; wrap the caller pointer yourself. | UMA Metal (when alignment allows zero-copy `newBufferWithBytesNoCopy`); CpuEngine (just reads central EValue). |

The two paths are mutually exclusive per vid. Pick one.

---

## 7. `upload_from_host` / `download_to_host` (DeviceEngine only)

Pure virtuals on `DeviceEngine`. Called by NativeBackend's TransferStep dispatch when a value crosses the host↔device boundary.

```cpp
Error upload_from_host(
    EValue& host_src_ev,
    EValue& dev_dst_ev,
    uint32_t dev_dst_value_id,
    Span<Event* const> wait_for,
    Event* signal);

Error download_to_host(
    EValue& dev_src_ev,
    uint32_t dev_src_value_id,
    EValue& host_dst_ev,
    Span<Event* const> wait_for,
    Event* signal);
```

### Contract

1. **Wait on dependencies.** Use `check_async_dependencies(wait_for, signal)` from `EngineUtils.h`. If any dep is Failed/Poisoned, poison `signal` and return `Error::Internal`.
2. **Resolve your buffer.** Look up `value_to_buffer_[dev_*_value_id]`.
3. **Propagate shape.** Update the destination tensor's shape to match the source. For static-shape engines, ET's `resize_tensor(dst_t, src_t.sizes())` (just mutates the TensorImpl) is enough because the buffer is already sized to fit. For dynamic-shape engines, call your own `Engine::resize_tensor(values, dev_*_value_id, src_t.sizes())` so the buffer is grown if needed before bytes move.
4. **Move bytes.** UMA path: if your buffer's host_ptr already equals the host EValue's pointer, no-op. Otherwise: `cudaMemcpyAsync` / `vkCmdCopyBuffer` / `memcpy` as appropriate.
5. **Signal.** `signal->prepare_signal(); signal->signal_complete();` on success. `signal->signal_failed(err)` on failure.

Use `SignalGuard` from `EngineUtils.h` to ensure the signal is settled even on early returns.

### What NOT to do

- Don't write the caller pointer into the central EValue here — that was already done by `bind_*`.
- Don't initiate transfers between two device runtimes. The host-canonical invariant means you only ever transfer to/from host. If a value needs to go from your runtime to another device runtime, the router emits two TransferSteps with host as the intermediate.

---

## 8. `resize_tensor` (dynamic shapes)

```cpp
Error resize_tensor(
    Span<EValue> values,
    uint32_t value_id,
    ArrayRef<aten::SizesType> new_sizes) override;
```

Mirrors ET's `resize_tensor` pattern lifted to the engine layer. The caller (typically the engine itself during shape inference, or a TransferStep about to write more bytes than fit) declares: "vid `value_id` needs to hold a tensor of shape `new_sizes`."

Default: `NotSupported`. Override only if your engine participates in dynamic shapes.

### Contract

After this returns Ok:
- `values[value_id].toTensor().sizes() == new_sizes`.
- The underlying Buffer can hold at least `sizeof(dtype) * numel(new_sizes)` bytes.
- The TensorImpl's `data_ptr` reflects the current storage location (may have changed if reallocation was required).

### Implementation pattern

```cpp
Error MyEngine::resize_tensor(
    Span<EValue> values,
    uint32_t value_id,
    ArrayRef<SizesType> new_sizes) {
  auto it = value_to_buffer_.find(value_id);
  if (it == value_to_buffer_.end()) return Error::InvalidArgument;

  auto& central_t = values[value_id].toTensor();
  size_t new_nbytes = bytes_for_sizes(central_t.scalar_type(), new_sizes);

  Buffer* buf = it->second;
  if (buf == nullptr || buf->size_bytes() < new_nbytes) {
    // Lazy alloc (buf == nullptr from deferred mem_obj_id < 0 case),
    // or grow (existing too small). Allocate fresh; the simplest
    // policy.
    buf = allocate_fresh_buffer(new_nbytes);
    value_to_buffer_[value_id] = buf;
  }

  ET_RETURN_IF_ERROR(::executorch::runtime::resize_tensor(central_t, new_sizes));
  if (void* hp = buf->host_ptr()) {
    central_t.unsafeGetTensorImpl()->set_data(hp);
  }
  return Error::Ok;
}
```

> **Memory-management caveat:** the simple pattern above leaves the old buffer owned by your engine until teardown — effectively leaking it until then. The reference implementations (CpuEngine, MetalEngine, HostPoolEngine) take this approach because dynamic-shape execution isn't yet exercised. For a production engine running long-lived inference with frequent shape changes, replace `allocate_fresh_buffer` + push-onto-owned-list with a recycling pool: free the previous buffer, reuse a slab, or otherwise bound the working set.

### When it's called

- **Producer kernel:** computes a dynamic output shape, calls `this->resize_tensor` on its own output vid before writing bytes.
- **TransferStep destination:** the destination engine resizes its buffer to match source shape before the byte move (handled inside `upload_from_host` / `download_to_host`).
- **Lazy first-use allocation:** for vids you claimed with `mem_obj_id < 0`, `resize_tensor` is the first time real storage gets allocated.

### Helpers in `EngineUtils.h`

- `bytes_for_sizes(dtype, sizes)` \u2014 compute byte count from dtype + sizes.
- `::executorch::runtime::resize_tensor(Tensor&, ArrayRef<SizesType>)` \u2014 mutate the TensorImpl's sizes/strides arrays.

---

## 9. `compile_segment`, `execute`, `wait`, `drain`, `make_event`

### `compile_segment`

```cpp
Result<CompiledSegment*> compile_segment(
    const Graph& graph,
    Span<const uint32_t> instruction_indices,
    Span<const uint32_t> input_value_ids,
    Span<const uint32_t> output_value_ids,
    Span<const std::pair<uint32_t, uint32_t>> value_remap);
```

Init-time. You receive the graph instruction indices for one segment plus an optional `value_remap` from the router (rewrites graph vids to mirror_ids your kernels should look up at execute time). Compile / encode whatever the runtime needs (ICB, pipeline, kernel handles). Return a `CompiledSegment*` that you own.

Non-compute engines (HostPool) return `Error::NotSupported` here — the router never assigns segments to them.

### `execute`

Per-execute hot path. Called with the segment, the central EValue array, wait_for events, and a signal event. Issue the work asynchronously, signal completion when done. Use the shape-on-event contract: by the time `signal` reaches `Complete`, every output's TensorImpl shape AND bound bytes must be valid.

Use `SignalGuard`. Refresh per op-arg: read `value_to_buffer_[vid]->host_ptr()` before invoking each kernel so kernels see the current pointer.

### `wait`

Block until the given Event reaches a terminal state. Default impl in `Event::wait_until_settled` is bounded spin + yield; CV-backed for true-async producers (see `CpuEvent`).

### `drain`

Block until all in-flight work issued by THIS engine reaches terminal state. Idempotent. Called at delegate teardown.

### `make_event`

Init-time factory. Return a runtime-specific `Event` subclass instance. CPU returns `CpuEvent`. Metal returns `MetalEvent`.

---

## 10. `upload_constants`

Init-time. You receive a list of `(value_id, ndm_key)` pairs. For each, materialize the constant from the NamedDataMap however you like:

- CPU: alias the FreeableBuffer's region zero-copy.
- UMA Metal: register region with MTLStream, alias zero-copy.
- Discrete GPU: allocate VRAM, copy bytes once.

Insert `(value_id → Buffer*)` into `value_to_buffer_`. Hold the FreeableBuffer alive (in your owned-buffers vector or a dedicated holder).

Constants do NOT use TransferStep. This call IS the upload path.

---

## 11. Threading & lifecycle summary

| Phase | When | Sync? |
|---|---|---|
| `compile_segment` | init | sync |
| `allocate_buffers` | init | sync |
| `upload_constants` | init | sync |
| `set_io_bindings` | init (after the above) | sync |
| `bind_inputs` / `bind_outputs` | every execute | sync; cheap |
| `resize_tensor` | as needed (init or per-execute) | sync; may allocate |
| `execute` | every execute | async (via signal) |
| `upload_from_host` / `download_to_host` | every execute | async (via signal) |
| `wait` | as needed | blocks |
| `drain` | teardown | blocks |
| destructor | teardown | sync; release everything |

---

## 12. Hard rules (the contract in one page)

1. **You own all your Buffers.** Allocate in `allocate_buffers` / `upload_constants` / `bind_*` / `resize_tensor`. Release in your destructor.
2. **NativeBackend tracks nothing about your storage.** The only cross-engine surface is the central EValue's `data_ptr`, which you write when you allocate host-resident bytes.
3. **Dispatch on `MemoryKind` alone.** No other field on `AllocRequest` gates behavior.
4. **Materialize TensorImpls for `DeviceMirror` vids yourself**, using `materialize_mirror_tensor()`. Store the `TensorImplStorage` for the lifetime of the engine.
5. **Self-filter in `bind_*`.** No bucketing happens outside the engine.
6. **Honor the host-canonical invariant.** All cross-runtime transfers go through host. Don't try to do peer-to-peer.
7. **Honor the shape-on-event contract.** Output TensorImpl shape AND bytes must be valid by the time `signal` reaches `Complete`.
8. **Settle every `signal` before returning** (via `SignalGuard` + `signal_complete` / `signal_failed`).
9. **`DeviceMirror` and `DeviceOnly` requests are non-negotiable.** You must Claim them — with one exception: if you can't honor a request (e.g., `mem_obj_id < 0` and you don't support `resize_tensor`), return `Declined` for that request. NativeBackend's `materialize_buffers` validates this and fails init.
10. **`mem_obj_id` has two regimes.** `>= 0` is planned (group, share, size from AOT). `< 0` is unplanned/dynamic (claim, defer to `resize_tensor`). Do not allocate at init for `mem_obj_id < 0`.
11. **If you support dynamic shapes, override `resize_tensor`.** Producer kernels (and TransferSteps that resize destinations) call this on you. The default `NotSupported` is acceptable **only if your engine also returns `Declined` (or fails the whole `allocate_buffers` batch) for any `DeviceMirror`/`DeviceOnly`/`HostMirror` request with `mem_obj_id < 0`.** Claiming a dynamic-mem_obj_id request without supporting `resize_tensor` is a contract violation — the deferred allocation will never happen.

---

## 13. Reference implementations

- **`runtimes/cpu/CpuEngine.{h,cpp}`** — host-resident DeviceEngine. Returns `handles_*_directly = true`, no `set_io_bindings` override. Implements `resize_tensor` for dynamic-shape vids.
- **`runtimes/metal/MetalEngine.{h,mm}`** — Apple Silicon UMA DeviceEngine. Default mirror behavior (`handles_*_directly = false`); `set_io_bindings` records mirror_ids; `bind_*` re-aliases MTLBuffers per execute. Implements `resize_tensor` via `stream_->allocator()`.
- **`runtimes/host_pool/HostPoolEngine.{h,cpp}`** — the Engine (not DeviceEngine). Handles `HostExtern` and `HostMirror`; supports `mem_obj_id < 0` deferred allocation via `resize_tensor`. Reference for nothing else; you won't write another HostPool.

Match the pattern from CpuEngine if your runtime is host-addressable. Match MetalEngine if your runtime has its own device memory and supports zero-copy UMA aliasing. Implement neither pattern yet for discrete GPU; the design supports it but no reference implementation exists.
