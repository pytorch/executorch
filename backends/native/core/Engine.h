/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Buffer.h>
#include <executorch/backends/native/core/Event.h>
#include <executorch/backends/native/core/MemoryKind.h>
#include <executorch/backends/native/core/RuntimeContext.h>
#include <executorch/backends/native/ir/GraphTypes.h> // reuse existing Graph

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>

namespace executorch {
namespace backends {
namespace native {

/**
 * Per-input/output binding (router output). Defined here so that Engine
 * can take Span<const InputBinding> in set_io_bindings without pulling
 * in Plan.h (Plan.h already depends on Engine.h).
 */
struct InputBinding {
  uint32_t value_id;
};

struct OutputBinding {
  uint32_t value_id;
};

/**
 * Per-program backend-private compiled state. Holds whatever the runtime
 * needs to dispatch a contiguous run of instructions: encoded ICB,
 * pipeline state objects, kernel handles, etc.
 *
 * Held by the Engine; referenced by ComputeStep.
 */
class CompiledSegment {
 public:
  virtual ~CompiledSegment() = default;
};

/**
 * Per-program execution state for one runtime. Owns the compiled segments
 * and the per-program buffers (allocated, IO-bound, and constant) plus
 * the per-engine value_id → Buffer table that resolves kernel-arg
 * storage at dispatch time.
 *
 * Ownership model:
 *   - The Engine owns every Buffer it allocated (intermediates, mirrors,
 *     constants, IO-bound). It releases all of them in its destructor.
 *     them all in its destructor.
 *   - NativeBackend never holds Buffer* directly. It tracks no per-vid
 *     storage state. All cross-engine coordination happens via the
 *     central EValue array (where engines write data_ptr for
 *     host-addressable buffers they own) and via value_id parameters
 *     on the per-execute APIs.
 *
 * All work-issuing methods are async-by-default. They return immediately
 * after enqueueing on the appropriate internal queue and signal the
 * provided Event* on completion. CPU runtime returns with the event
 * already complete; per-call overhead is two branches.
 *
 * See §4.7 of the design doc.
 */
class Engine {
 public:
  // Per-program construction. Engine stores the Graph reference for its
  // lifetime; the orchestrator (NativeBackend / DelegateInstance) MUST
  // keep the Graph alive at least as long as this Engine. NativeBackend's
  // DelegateInstance enforces this by declaring its Engines AFTER the
  // Graph member so Engines are torn down first (reverse declaration
  // order in the destructor).
  explicit Engine(const ::executorch::backends::portable::Graph& graph)
      : graph_(graph) {}
  virtual ~Engine() = default;

  // Read-only accessor for the engine's program graph. Available from
  // construction onward; usable in any subsequent method (including
  // pre-route hooks like handles_*_directly).
  const ::executorch::backends::portable::Graph& graph() const {
    return graph_;
  }

  // Lifecycle of an Engine instance (per delegate program):
  //   1. Runtime::instantiate(graph)   — Engine constructed; graph_
  //                                      stored and valid for the
  //                                      engine's lifetime.
  //   2. handles_*_directly(vid)       — called by router during route(),
  //                                      pre-compile_segment. Engine may
  //                                      consult graph() for per-vid
  //                                      policy (dtype, dynamism, etc.).
  //   3. compile_segment               — one per kernel segment.
  //   4. allocate_buffers              — post-route.
  //   5. upload_constants              — post-route, after allocate_buffers.
  //   6. set_io_bindings               — post-route, after upload_constants.
  //   7. bind_inputs / bind_outputs    — per-execute.
  //   8. execute                       — per-execute.
  //   9. wait                          — per-execute (or any time after 6).
  //  10. drain                         — at engine destruction.
  //
  // Capability discipline at the API surface:
  //   graph()             — read-only schema (immutable AOT metadata).
  //                         Never gives access to runtime EValue state.
  //   Span<EValue> values — mutable runtime state; passed only to
  //                         methods that legitimately read/write it
  //                         (allocate_buffers, bind_*, execute,
  //                         resize_tensor). Methods that don't take it
  //                         are non-mutating w.r.t. the central array.

  // Compile a contiguous run of instructions. Encodes ICB / shaders /
  // pipelines as appropriate. Returned CompiledSegment* is owned by
  // Engine. SYNCHRONOUS (init-time only). Graph reached via graph().
  //
  // value_remap: optional rewrite from "graph value_id" to "value_id this
  // segment should look up in the engine's internal value→Buffer table."
  // Used when the router creates new value_ids for cross-runtime mirror
  // destinations: the segment's kernels were exported referencing the
  // original value_id V, but the router needs them to read from V' (the
  // engine's mirror Buffer). The mapping is applied per op-arg at execute
  // time. Pass an empty Span for single-runtime / no-rewrite case.
  virtual ::executorch::runtime::Result<CompiledSegment*> compile_segment(
      ::executorch::runtime::Span<const uint32_t> instruction_indices,
      ::executorch::runtime::Span<const uint32_t> input_value_ids,
      ::executorch::runtime::Span<const uint32_t> output_value_ids,
      ::executorch::runtime::Span<const std::pair<uint32_t, uint32_t>>
          value_remap) = 0;

  // Sentinel: an invalid / unused value_id. Used by AllocRequest's
  // host_mirror_value_id to mean "no host-side partner."
  static constexpr uint32_t kInvalidValueId =
      std::numeric_limits<uint32_t>::max();

  struct AllocRequest {
    uint32_t value_id; // index into `values` for this request

    // AOT memory-planner slot id. Two regimes:
    //   mem_obj_id >= 0 : Planned. Multiple requests sharing the same
    //                     id MAY share storage (the planner determined
    //                     their lifetimes don't overlap). The engine
    //                     groups by id and sizes one Buffer per group.
    //   mem_obj_id <  0 : Unplanned. No AOT-known size; the vid is
    //                     dynamic-shaped or unbounded. The engine
    //                     defers allocation: claim the request without
    //                     allocating storage; allocate lazily on the
    //                     first resize_tensor() call (or first use).
    int32_t mem_obj_id;

    // Addressing contract this allocation must satisfy. The router
    // stamps this on every emitted request; the backend dispatches on
    // this enum alone (no other gating fields):
    //
    //   HostExtern    : caller-owned host storage. Only HostPoolEngine
    //                   handles it (allocates a thin Aliasing wrapper
    //                   re-pointed per execute by bind_inputs /
    //                   bind_outputs). Used for graph IO. Not biddable.
    //   HostMirror    : host side of a mirror pair. Bid-eligible — a
    //                   device engine may claim it (e.g., CUDA pinned
    //                   host via cudaMallocHost; Metal Shared on UMA);
    //                   HostPool is the fallback claimant.
    //   DeviceMirror  : device side of a mirror pair. Allocated by the
    //                   targeted device engine. The companion
    //                   HostMirror is referenced by
    //                   host_mirror_value_id; the engine reads
    //                   values[host_mirror_value_id].toTensor()
    //                   .data_ptr() to find the host pointer (set by
    //                   whichever engine claimed the host half).
    //                   The provider MAY collapse the pair into one
    //                   physical allocation (UMA) by aliasing the
    //                   host-side buffer's host_ptr (zero-copy) when
    //                   that pointer is non-null.
    //   DeviceOnly    : owned-by-this-runtime allocation. The provider
    //                   chooses the allocator (e.g., cudaMalloc,
    //                   MTLPrivate). No host contract; host_ptr() may
    //                   be null.
    MemoryKind kind = MemoryKind::HostExtern;

    // For DeviceMirror requests: the value_id of the host-side partner
    // in the mirror pair. The engine reads
    // values[host_mirror_value_id].toTensor().data_ptr() at allocate
    // time to find the host pointer to alias against (UMA collapse) or
    // remember (discrete-GPU staging copy).
    //
    // kInvalidValueId means "no host partner," typical for DeviceOnly.
    uint32_t host_mirror_value_id = kInvalidValueId;
  };

  // Capability comment for HostMirror/HostExtern below.

  enum class AllocClaim : uint8_t {
    Claimed, // engine owns this buffer; populated values[vid] if
             // host-addressable (writes data_ptr for kernels to read).
    Declined, // engine refuses; only valid for HostMirror/HostExtern
              // (HostPool will claim). For DeviceMirror/DeviceOnly,
              // declining is a router/init bug.
  };

  // Bid auction over storage requests. Engine claims requests it wants
  // and declines others. Today only HostMirror is biddable (engines may
  // claim, HostPool floors); HostExtern is HostPool-only; DeviceMirror /
  // DeviceOnly are dispatched only to the targeted device engine and
  // must be Claimed.
  //
  // For each Claimed request:
  //   - The engine internally allocates a Buffer and inserts
  //     (value_id → Buffer*) into its internal value→Buffer table.
  //   - If the resulting Buffer is host-addressable
  //     (`buf->host_ptr() != nullptr`), the engine writes
  //     values[value_id].toTensor().unsafeGetTensorImpl()->set_data(host_ptr).
  //     This is how cross-engine consumers (other engines, NativeBackend
  //     for diagnostics) discover the host pointer for a vid: they read
  //     values[vid].toTensor().data_ptr().
  //   - For DeviceMirror requests, the engine reads
  //     values[req.host_mirror_value_id].toTensor().data_ptr() to find
  //     the host partner's pointer for UMA aliasing (if non-null) or
  //     staging-copy bookkeeping (if null).
  //
  // out_claims is parallel to requests; engine writes one entry per
  // request. Engines must process requests strictly in the order
  // presented (so a HostMirror claim later can be informed by a
  // DeviceMirror claim earlier in the same batch, etc.).
  //
  // SYNCHRONOUS (init-time only).
  // values is passed first by convention (mutable runtime state — the
  // capability — leads); requests is the per-call input list; out_claims
  // is the engine's accept/decline reply.
  virtual ::executorch::runtime::Error allocate_buffers(
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<const AllocRequest> requests,
      ::executorch::runtime::Span<AllocClaim> out_claims) = 0;

  // Per-execute IO binding. Called by NativeBackend on every engine,
  // ONCE per execute (not bucketed). The engine self-filters via its
  // internal IO bindings table built at init-time by set_io_bindings.
  //
  // input_args[i] is the caller's EValue for graph input i (same
  // ordering as Plan::inputs). The engine walks its stored bindings
  // to find which (graph_idx, internal_vid) pairs to act on.
  //
  // Default: empty internal bindings table -> no-op. Engines that own
  // any IO override these.
  //
  // INPUT IMMUTABILITY: input_args is Span<const EValue* const> —
  // both the array of pointers and the EValues they point to are
  // const. Engines may read caller_t.const_data_ptr() / .nbytes() /
  // .sizes() but MUST NOT mutate the caller's input Tensor's data,
  // shape, or storage. PyTorch contract: model inputs are immutable
  // from the engine's view. The pointer obtained from a const Tensor's
  // mutable_data_ptr() (which is const-callable in ET because it
  // doesn't mutate the Tensor's own state) is used only for aliasing
  // (the kernel reads through the alias) or for memcpy-from-source in
  // pool-fallback paths — never for memcpy-into-source.
  virtual ::executorch::runtime::Error bind_inputs(
      ::executorch::runtime::Span<::executorch::runtime::EValue> /*values*/,
      ::executorch::runtime::Span<const ::executorch::runtime::EValue* const>
      /*input_args*/) {
    return ::executorch::runtime::Error::Ok;
  }

  // Symmetric to bind_inputs for graph outputs.
  virtual ::executorch::runtime::Error bind_outputs(
      ::executorch::runtime::Span<::executorch::runtime::EValue> /*values*/,
      ::executorch::runtime::Span<::executorch::runtime::EValue* const>
      /*output_args*/) {
    return ::executorch::runtime::Error::Ok;
  }

  // Per-program IO setup. Called once after allocate_buffers and
  // upload_constants. Each engine inspects the graph IO lists, decides
  // which it owns (via its own claimed AllocRequests + its
  // handles_*_directly responses), and stores its internal IO bindings
  // table. Cross-engine bookkeeping does not exist; each engine is
  // self-sufficient.
  //
  // Default: no-op (engine owns no IO bindings, e.g., CpuEngine which
  // resolves graph IO via the central EValue's data_ptr at execute
  // time).
  virtual ::executorch::runtime::Error set_io_bindings(
      ::executorch::runtime::Span<const InputBinding> /*graph_inputs*/,
      ::executorch::runtime::Span<const OutputBinding> /*graph_outputs*/) {
    return ::executorch::runtime::Error::Ok;
  }

  // Capability queries: does this engine handle the caller's pointer
  // for a graph IO vid directly, instead of asking the router to mint
  // a separate device-side mirror? Called per (engine, graph IO vid)
  // at route time. The two paths are mutually exclusive per vid.
  //
  // Return false (default — router mints a mirror):
  //   - Router emits a DeviceMirror AllocRequest on this engine paired
  //     to the host-side HostExtern via host_mirror_value_id.
  //   - Router emits a per-execute TransferStep (upload for inputs;
  //     download for outputs) that moves bytes between the host
  //     wrapper and the device mirror.
  //   - The compiled segment references the mirror_id (via value_remap).
  //   - bind_inputs / bind_outputs is NOT called on this engine for
  //     this vid; the TransferStep handles per-execute byte movement.
  //
  // Return true (engine wraps caller pointer directly):
  //   - No DeviceMirror is minted; no TransferStep is emitted.
  //   - bind_inputs / bind_outputs IS called on this engine for this
  //     vid per execute; the engine wraps the caller's pointer into
  //     its own Buffer view (e.g., MTLBuffer via
  //     newBufferWithBytesNoCopy on UMA, with whatever alignment
  //     fallback the engine needs).
  //
  // Use cases:
  //   - CpuEngine / UMA Metal: return true when the caller's pointer
  //     is directly consumable (zero-copy alias of host bytes).
  //   - Discrete GPU (CUDA, Vulkan, Intel Mac Metal): return false
  //     (must copy bytes into device-owned VRAM via TransferStep).
  virtual bool handles_input_directly(uint32_t /*graph_input_value_id*/) const {
    return false;
  }

  virtual bool handles_output_directly(
      uint32_t /*graph_output_value_id*/) const {
    return false;
  }

  // Materialize one or more graph constants on this runtime;
  // persistent across executes. The Engine reads from the NamedDataMap
  // directly so it can choose how to materialize:
  //  - CPU: HostBuffer aliases the FreeableBuffer's region (zero-copy).
  //  - Apple-Silicon Metal: registers the FreeableBuffer's region with
  //    MetalStream; MetalBuffer aliases (zero-copy).
  //  - Discrete GPU: allocate device buffer + load_data_into directly
  //    (avoids host roundtrip).
  //
  // The Engine owns each resulting Buffer AND the underlying
  // FreeableBuffer (so the mmap'd region stays alive as long as the
  // Buffer aliases it). The engine inserts (value_id → Buffer*) into
  // its internal table; nothing leaves through this API.
  //
  // SYNCHRONOUS (init-time only). Constants do NOT go through
  // upload_from_host — they have their own dedicated path.
  //
  // Driven by NativeBackend post-route from Plan::const_plans (the
  // router emits per-provider ConstRequest lists during planning;
  // route() never calls this method). Symmetric to allocate_buffers /
  // alloc_plans.
  // Constants come from one of two sources, mutually exclusive:
  //   1. NDM-stored: ndm_key is non-empty; engine fetches bytes via
  //      ndm.get_data(ndm_key). Used for named parameters/buffers
  //      that the AOT externalized.
  //   2. Inline: ndm_key is empty AND inline_data is non-empty; the
  //      bytes live in the program's constant_buffer flatbuffer
  //      region, kept alive by the Module's processed FreeableBuffer.
  //      Used for AOT-lifted constants (e.g., literals) that ET keeps
  //      "close to code" rather than promoting to NDM.
  // The engine aliases the source bytes (zero-copy) for host-resident
  // runtimes; for discrete-VRAM runtimes it copies once into device
  // memory.
  struct ConstRequest {
    uint32_t value_id; // graph value to bind to
    std::string_view ndm_key; // NDM key, or empty for inline
    ::executorch::runtime::Span<const uint8_t>
        inline_data; // inline bytes, or empty
  };

  virtual ::executorch::runtime::Error upload_constants(
      const ::executorch::runtime::NamedDataMap* ndm,
      ::executorch::runtime::Span<const ConstRequest> requests) = 0;

  // Provide the storage backing one Event slot. Called once per slot at
  // Plan construction. const: just constructs an Event; no engine-state
  // mutation.
  virtual std::unique_ptr<Event> make_event() const = 0;

  //=== Async work issuance + sync helpers ===================================
  //
  // Public hot-path API: `execute` (intra-runtime kernel dispatch). The
  // host<->device transfer pair (`upload_from_host` / `download_to_host`)
  // lives on DeviceEngine (see below); the host-canonical invariant
  // means HostPool never needs them.

  // values: the DelegateInstance's EValue array (carries dtype/shape and
  // scalar inputs). The engine looks up tensor storage backings from its
  // own internal value→Buffer table (populated by allocate_buffers,
  // upload_constants, bind_inputs, bind_outputs).
  //
  // Engines typically refresh values[vid].toTensor().data_ptr from
  // their internal Buffer's host_ptr() per op-arg before kernel dispatch
  // (single source of truth).
  //
  // SHAPE-ON-EVENT CONTRACT: by the time `signal` reaches
  // EventStatus::Complete, every output value's TensorImpl shape AND
  // bound Buffer bytes MUST be valid for downstream consumers.
  // Backends that determine output shapes synchronously inside execute()
  // (CPU portable kernels, Metal-with-MPSGraph metadata) update shape
  // before encoding the kernel. Backends whose output shapes are only
  // known after the GPU runs must register a completion handler that
  // updates the output TensorImpls' shape arrays AND THEN signals the
  // event last.
  //
  // segment is engine-owned and immutable post-compile_segment.
  virtual ::executorch::runtime::Error execute(
      const CompiledSegment* segment,
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) = 0;

  // Block CPU until event signals. Returns Error::Ok iff event reaches
  // EventStatus::Complete; returns the stored error if Failed/Poisoned.
  // const: doesn't mutate engine state — just polls/waits the event.
  virtual ::executorch::runtime::Error wait(Event* event) const = 0;

  // Stable per-Engine id within this Engine's RuntimeContext.
  virtual InstanceId id() const = 0;

  // Update the shape (and underlying storage capacity) of a vid this
  // engine claims. Mirrors ET's resize_tensor pattern lifted to the
  // engine layer.
  //
  // Parameters:
  //   values    : the central DelegateInstance EValue array. The engine
  //               reads values[value_id]'s TensorImpl for dtype/dim_order
  //               and updates its shape and data_ptr.
  //   value_id  : vid this engine owns.
  //   new_sizes : the desired tensor shape.
  //
  // After this returns Ok:
  //   values[value_id].toTensor().sizes() == new_sizes.
  //   The underlying Buffer can hold at least
  //   sizeof(dtype) * numel(new_sizes) bytes.
  //   data_ptr on the TensorImpl reflects current storage location
  //   (may have changed if reallocation was required).
  //
  // Use cases:
  //   - Producer kernel determines a dynamic output shape and calls
  //     this on its own engine before writing.
  //   - TransferStep upload/download: destination engine resizes its
  //     buffer to match source shape before the byte move.
  //   - Lazy allocation: a vid with mem_obj_id < 0 (unplanned, no AOT
  //     bound) is allocated on first resize_tensor call.
  //
  // Default: NotSupported. Engines that participate in dynamic shapes
  // must override.
  virtual ::executorch::runtime::Error resize_tensor(
      ::executorch::runtime::Span<::executorch::runtime::EValue> /*values*/,
      uint32_t /*value_id*/,
      ::executorch::runtime::ArrayRef<::executorch::aten::SizesType>
      /*new_sizes*/) {
    return ::executorch::runtime::Error::NotSupported;
  }

  // Drain in-flight work *issued by this Engine only*. Blocks until
  // every currently outstanding submission this Engine has issued via
  // copy_*/execute reaches a terminal state. Idempotent.
  virtual void drain() = 0;

 protected:
  // Stored at construction; valid for the engine's lifetime. NEVER null.
  // Reference (not pointer) so the type system enforces "engine always
  // has a graph" — derived classes must initialize via base ctor.
  const ::executorch::backends::portable::Graph& graph_;
};

/**
 * DeviceEngine adds the host<->device transfer methods. By the
 * host-canonical invariant, every TransferStep has either src or dst
 * on the host runtime; the non-host side is always a DeviceEngine.
 *
 * NativeBackend's TransferStep dispatch casts the non-host engine to
 * DeviceEngine* and calls the appropriate direction. HostPoolEngine
 * inherits Engine directly (it never runs as a device).
 */
class DeviceEngine : public Engine {
 public:
  using Engine::Engine; // inherit explicit Engine(const Graph&) ctor

  // Make the engine's Buffer for `dev_dst_value_id` reflect host_src_ev's
  // data_ptr by the time `signal` reaches Complete. The engine resolves
  // its own Buffer internally from dev_dst_value_id; the host pointer
  // is read from host_src_ev.toTensor().data_ptr().
  //
  // Implementation strategy is the runtime's call:
  //   - Host-addressable runtimes (CPU, Apple-Silicon Metal) typically
  //     see "alias unchanged" (the bound Buffer already points at the
  //     host source's storage from allocate_buffers / bind_inputs).
  //     Returns immediately after signaling.
  //   - Discrete GPU (Vulkan, CUDA): real copy from host_ptr into
  //     pre-allocated VRAM.
  //
  // Also propagates shape from src to dst's TensorImpl before signaling
  // (per the shape-on-event contract).
  virtual ::executorch::runtime::Error upload_from_host(
      const ::executorch::runtime::EValue& host_src_ev,
      ::executorch::runtime::EValue& dev_dst_ev,
      uint32_t dev_dst_value_id,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) = 0;

  // Symmetric: make host_dst_ev's data_ptr reflect the engine's Buffer
  // for `dev_src_value_id`. Engine resolves its own Buffer internally.
  virtual ::executorch::runtime::Error download_to_host(
      const ::executorch::runtime::EValue& dev_src_ev,
      uint32_t dev_src_value_id,
      ::executorch::runtime::EValue& host_dst_ev,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) = 0;
};

} // namespace native
} // namespace backends
} // namespace executorch
