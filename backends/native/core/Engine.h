/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Buffer.h>
#include <executorch/backends/native/core/BufferRole.h>
#include <executorch/backends/native/core/Event.h>
#include <executorch/backends/native/ir/GraphTypes.h> // reuse existing Graph
#include <executorch/backends/native/core/MemoryKind.h>
#include <executorch/backends/native/core/RuntimeContext.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <string_view>

namespace executorch {
namespace backends {
namespace native {

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
 *   - NativeBackend never holds Buffer* directly. It tracks one piece of
 *     cross-engine state: value_owner[vid] → RuntimeIndex (which engine
 *     claimed this vid). All other coordination happens via the EValue
 *     array (where engines write data_ptr for host-addressable buffers
 *     they own) and via value_id parameters on the per-execute APIs.
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
  virtual ~Engine() = default;

  // Compile a contiguous run of instructions. Encodes ICB / shaders /
  // pipelines as appropriate. Returned CompiledSegment* is owned by
  // Engine. SYNCHRONOUS (init-time only).
  //
  // value_remap: optional rewrite from "graph value_id" to "value_id this
  // segment should look up in the engine's internal value→Buffer table."
  // Used when the router creates new value_ids for cross-runtime mirror
  // destinations: the segment's kernels were exported referencing the
  // original value_id V, but the router needs them to read from V' (the
  // engine's mirror Buffer). The mapping is applied per op-arg at execute
  // time. Pass an empty Span for single-runtime / no-rewrite case.
  virtual ::executorch::runtime::Result<CompiledSegment*> compile_segment(
      const ::executorch::backends::portable::Graph& graph,
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
    int32_t mem_obj_id; // AOT memory-planner slot id. Multiple requests
                        // sharing the same id MAY share storage (the
                        // planner determined their lifetimes don't
                        // overlap). The field's semantics are pure
                        // grouping; "is this graph IO?" is in `role`.

    // Addressing contract this allocation must satisfy. The router
    // stamps this on every emitted request; the backend interprets
    // it to choose an allocation strategy:
    //
    //   HostOnly      : plain host allocation. Only HostPoolEngine is
    //                   asked for this.
    //   HostMirror    : host side of a mirror pair. Bid-eligible — a
    //                   device engine may claim it (e.g., CUDA pinned
    //                   host via cudaMallocHost; Metal Shared on UMA);
    //                   HostPool is the fallback claimant.
    //   DeviceMirror  : device side of a mirror pair. Allocated by the
    //                   compute provider the router targeted. The
    //                   companion HostMirror is referenced by
    //                   host_mirror_value_id; the engine reads
    //                   values[host_mirror_value_id].toTensor()
    //                   .data_ptr() to find the host pointer (set by
    //                   whichever engine claimed the host half).
    //                   The provider MAY collapse the pair into one
    //                   physical allocation (UMA) by aliasing the
    //                   host-side buffer's host_ptr (zero-copy) when
    //                   that pointer is non-null and the engine claimed
    //                   the HostMirror request too.
    //   DeviceOnly    : owned-by-this-runtime allocation. The provider
    //                   chooses the allocator (e.g., cudaMalloc,
    //                   MTLPrivate). No host contract; host_ptr() may
    //                   be null.
    MemoryKind kind = MemoryKind::HostOnly;

    // Why this allocation exists (orthogonal to addressing). Today's
    // model: graph IO never reaches allocate_buffers — IO is handled
    // exclusively via per-execute bind_inputs/bind_outputs. role
    // typically equals BufferRole::Internal for AllocRequests; Constant
    // requests are used for constants if needed. Kept as a field for
    // diagnostics and future extensibility.
    BufferRole role = BufferRole::Internal;

    // For DeviceMirror requests: the value_id of the host-side partner
    // in the mirror pair. The engine reads
    // values[host_mirror_value_id].toTensor().data_ptr() at allocate
    // time to find the host pointer to alias against (UMA collapse) or
    // remember (discrete-GPU staging copy).
    //
    // kInvalidValueId means "no host partner," typical for DeviceOnly.
    uint32_t host_mirror_value_id = kInvalidValueId;
  };

  // Returned per-request from allocate_buffers. The engine reports
  // whether it claimed each request; declined requests fall through to
  // the next bidder (HostPool is the floor for HostMirror/HostOnly).
  enum class AllocClaim : uint8_t {
    Claimed,    // engine owns this buffer; populated values[vid] if
                // host-addressable (writes data_ptr for kernels to read).
    Declined,   // engine refuses; only valid for HostMirror/HostOnly
                // (HostPool will claim). For DeviceMirror/DeviceOnly,
                // declining is a router/init bug.
  };

  // Bid auction over storage requests. Engine claims requests it wants
  // and declines others.
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
  virtual ::executorch::runtime::Error allocate_buffers(
      ::executorch::runtime::Span<const AllocRequest> requests,
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<AllocClaim> out_claims) = 0;

  // Per-execute IO binding. NativeBackend buckets the graph's input
  // values per owning engine (using value_owner) and calls bind_inputs
  // once per engine with that engine's bucket.
  //
  // Parameters (parallel spans):
  //   values        : the central DelegateInstance EValue array (mutable;
  //                   engine updates values[value_ids[i]]: resizes the
  //                   TensorImpl shape to caller_evs[i].sizes(); sets
  //                   data_ptr to caller's pointer for host-addressable
  //                   buffers).
  //   caller_evs[i] : caller's EValue (carrying real data_ptr / shape)
  //                   that drives binding for value_ids[i].
  //   value_ids[i]  : the vid in this engine's namespace (the host vid
  //                   if this is the host-pool / host-claiming engine,
  //                   or a mirror_id if this is a device engine that
  //                   owns a mirror of the graph input).
  //
  // The engine internally:
  //   1. Re-aliases (or allocates fresh) its Buffer for value_ids[i] to
  //      caller's storage. Updates its internal value→Buffer table.
  //   2. Resizes values[value_ids[i]] TensorImpl to caller's shape.
  //   3. If the resulting Buffer is host-addressable, sets data_ptr on
  //      values[value_ids[i]] TensorImpl to caller's pointer.
  //
  // Default returns NotImplemented; engines that own no IO bindings
  // for any vid (i.e., never receive a non-empty bucket) inherit the
  // default safely.
  virtual ::executorch::runtime::Error bind_inputs(
      ::executorch::runtime::Span<::executorch::runtime::EValue> /*values*/,
      ::executorch::runtime::Span<const ::executorch::runtime::EValue>
          /*caller_evs*/,
      ::executorch::runtime::Span<const uint32_t> /*value_ids*/) {
    return ::executorch::runtime::Error::NotImplemented;
  }

  // Symmetric to bind_inputs for graph outputs.
  virtual ::executorch::runtime::Error bind_outputs(
      ::executorch::runtime::Span<::executorch::runtime::EValue> /*values*/,
      ::executorch::runtime::Span<const ::executorch::runtime::EValue>
          /*caller_evs*/,
      ::executorch::runtime::Span<const uint32_t> /*value_ids*/) {
    return ::executorch::runtime::Error::NotImplemented;
  }

  // Capability queries: should the router mint an engine-side mirror
  // for a graph input / graph output that this engine touches? Called
  // per (engine, graph IO vid) at route time.
  //
  // Return true (default):
  //   - Router mints a DeviceMirror for this (vid, engine) pair.
  //   - AllocPlanner emits a DeviceMirror AllocRequest on this engine
  //     paired back to the host vid via host_mirror_value_id.
  //   - TransferPlanner emits a pre-segment upload (input) and/or
  //     post-segment download (output) TransferStep.
  //   - The compiled segment references the mirror_id (via value_remap).
  //
  // Return false:
  //   - No DeviceMirror is minted; no upload/download TransferStep is
  //     emitted; the compiled segment references the original IO vid
  //     directly.
  //   - The host pool's IO Buffer (re-aliased per execute by the host
  //     pool's bind_inputs/bind_outputs to the caller's data_ptr) is
  //     the only allocation for this IO.
  //   - The engine takes responsibility for resolving the original vid
  //     at execute time (typically by reading values[vid].toTensor()
  //     .data_ptr() and wrapping the host pointer into its own Buffer
  //     view, with whatever alignment fallback the engine needs).
  //
  // Defaults to true for safety. Engines opt out per IO once they've
  // wired up the execute-side resolve.
  //
  // Use cases:
  //   - CpuEngine / UMA Metal: may opt out when the caller's pointer
  //     is directly consumable (zero-copy alias of host bytes).
  //   - Discrete GPU (CUDA, Vulkan, Intel Mac Metal): always opts in
  //     (must copy bytes into device-owned VRAM).
  virtual bool wants_input_mirror(uint32_t /*graph_input_value_id*/) const {
    return true;
  }

  virtual bool wants_output_mirror(uint32_t /*graph_output_value_id*/) const {
    return true;
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
  struct ConstRequest {
    uint32_t value_id;            // graph value to bind to
    std::string_view ndm_key;     // key in the NamedDataMap
  };

  virtual ::executorch::runtime::Error upload_constants(
      const ::executorch::runtime::NamedDataMap& ndm,
      ::executorch::runtime::Span<const ConstRequest> requests) = 0;

  // Provide the storage backing one Event slot. Called once per slot at
  // Plan construction.
  virtual std::unique_ptr<Event> make_event() = 0;

  //=== Async work issuance + sync helpers ===================================
  //
  // Public hot-path API: `execute` (intra-runtime kernel dispatch) and the
  // host↔device pair `upload_from_host` / `download_to_host` (cross-runtime
  // moves between CPU and a device runtime).
  //
  // The transfer methods only ever live on the **device** Engine:
  //   * upload_from_host:  CPU produced a value, this device consumes it.
  //   * download_to_host:  this device produced a value, CPU consumes it.
  // CpuEngine / HostPoolEngine never have these called against them when
  // the device side is non-host (NativeBackend dispatches to the device
  // engine). HostPool does override them for host↔host fallback paths.
  //
  // We never need a runtime↔runtime path because v1 has at most one
  // non-host runtime. Any transfer between two non-host runtimes (future)
  // would route through host as two steps.

  // Make the engine's Buffer for `dev_dst_value_id` reflect host_src_ev's
  // data_ptr by the time `signal` reaches Complete. The engine resolves
  // its own Buffer internally from dev_dst_value_id; the host pointer
  // is read from host_src_ev.toTensor().mutable_data_ptr().
  //
  // Implementation strategy is the runtime's call:
  //   - Host-addressable runtimes (CPU, Apple-Silicon Metal) typically
  //     see "alias unchanged" (the bound Buffer already points at the
  //     host source's storage from allocate_buffers / bind_inputs).
  //     Returns immediately after signaling.
  //   - Discrete GPU (Vulkan): real copy from host_ptr into pre-
  //     allocated VRAM via vkCmdCopyBuffer.
  //
  // Also propagates shape from src to dst's TensorImpl before signaling
  // (per the shape-on-event contract).
  //
  // Default: NotImplemented. Override in non-host engines.
  virtual ::executorch::runtime::Error upload_from_host(
      ::executorch::runtime::EValue& /*host_src_ev*/,
      ::executorch::runtime::EValue& /*dev_dst_ev*/,
      uint32_t /*dev_dst_value_id*/,
      ::executorch::runtime::Span<Event* const> /*wait_for*/,
      Event* /*signal*/) {
    return ::executorch::runtime::Error::NotImplemented;
  }

  // Symmetric: make host_dst_ev's data_ptr reflect the engine's Buffer
  // for `dev_src_value_id`. Engine resolves its own Buffer internally.
  //
  // Default: NotImplemented. Override in non-host engines.
  virtual ::executorch::runtime::Error download_to_host(
      ::executorch::runtime::EValue& /*dev_src_ev*/,
      uint32_t /*dev_src_value_id*/,
      ::executorch::runtime::EValue& /*host_dst_ev*/,
      ::executorch::runtime::Span<Event* const> /*wait_for*/,
      Event* /*signal*/) {
    return ::executorch::runtime::Error::NotImplemented;
  }

  // Inputs guaranteed on this runtime; outputs MUST end up here.
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
  virtual ::executorch::runtime::Error execute(
      CompiledSegment* segment,
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) = 0;

  // Block CPU until event signals. Returns Error::Ok iff event reaches
  // EventStatus::Complete; returns the stored error if Failed/Poisoned.
  virtual ::executorch::runtime::Error wait(Event* event) = 0;

  // Stable per-Engine id within this Engine's RuntimeContext.
  virtual InstanceId id() const = 0;

  // Drain in-flight work *issued by this Engine only*. Blocks until
  // every currently outstanding submission this Engine has issued via
  // copy_*/execute reaches a terminal state. Idempotent.
  virtual void drain() = 0;
};

} // namespace native
} // namespace backends
} // namespace executorch
