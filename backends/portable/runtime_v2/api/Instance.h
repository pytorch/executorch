/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/BindingTable.h>
#include <executorch/backends/portable/runtime_v2/api/Buffer.h>
#include <executorch/backends/portable/runtime_v2/api/Event.h>
#include <executorch/backends/portable/runtime_v2/api/RuntimeContext.h>
#include <executorch/backends/portable/runtime_v2/api/GraphTypes.h>  // reuse existing Graph

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Per-program backend-private compiled state. Holds whatever the runtime
 * needs to dispatch a contiguous run of instructions: encoded ICB,
 * pipeline state objects, kernel handles, etc.
 *
 * Held by the Instance; referenced by ComputeStep.
 */
class CompiledSegment {
 public:
  virtual ~CompiledSegment() = default;
};

/**
 * Per-program execution state for one runtime. Owns the compiled segments
 * and the per-program buffers.
 *
 * All work-issuing methods are async-by-default. They return immediately
 * after enqueueing on the requested QueueKind and signal the provided
 * Event* on completion. CPU runtime returns with the event already
 * complete; per-call overhead is two branches.
 *
 * See §4.7 of the design doc.
 */
class Instance {
 public:
  virtual ~Instance() = default;

  // Compile a contiguous run of instructions. Encodes ICB / shaders /
  // pipelines as appropriate. Returned CompiledSegment* is owned by
  // Instance.
  // SYNCHRONOUS (init-time only).
  //
  // value_remap: optional rewrite from "graph value_id" to "value_id this
  // segment should look up in the BindingTable." Used when the router
  // synthesizes new value_ids for cross-runtime transfer destinations:
  // the segment's kernels were exported referencing the original value_id
  // V, but the router needs them to read from V' (a destination Buffer
  // on this runtime). The mapping is applied per op-arg at execute time.
  // Pass an empty Span for single-runtime / no-rewrite case.
  virtual ::executorch::runtime::Result<CompiledSegment*> compile_segment(
      const ::executorch::backends::portable::Graph& graph,
      ::executorch::runtime::Span<const uint32_t> instruction_indices,
      ::executorch::runtime::Span<const uint32_t> input_value_ids,
      ::executorch::runtime::Span<const uint32_t> output_value_ids,
      ::executorch::runtime::Span<const std::pair<uint32_t, uint32_t>>
          value_remap) = 0;

  // Batched allocation request. The router builds a list of these (one
  // per value_id this Instance is asked to back) and calls allocate_all
  // once at init.
  //
  // The backend reads the tensor's metadata (dtype, sizes, dim_order)
  // from `values[req.value_id].toTensor()`. Backends own all memory
  // planning internally:
  //   - CPU / Apple Silicon Metal: typically loop and allocate per-request.
  //   - Vulkan: group by mem_obj_id, aggregate VkMemoryRequirements (max
  //     size, max alignment, AND'd memoryTypeBits), allocate one VMA
  //     allocation per group, bind each user's VkBuffer to it.
  //
  // Released at Plan tear-down via release_buffer().
  struct AllocRequest {
    uint32_t value_id;     // index into `values` for this request
    int32_t mem_obj_id;    // -1 = dedicated allocation; else share with same id

    // Set ONLY for cross-runtime synthetic values (segment IO between
    // segments on different providers). Carries the source value's
    // Buffer — the host-side end of the transfer. Since all cross-
    // runtime transfers go through host, segment IO always has a host
    // buffer.
    //
    // If host_alias != nullptr AND host_alias->host_ptr() is non-null,
    // the backend MAY back this allocation by aliasing
    // host_alias->host_ptr() (zero-copy). If host_alias is null
    // (non-synthetic) or host_alias->host_ptr() is null (source is
    // VRAM-only, e.g., Vulkan), the backend allocates fresh and
    // per-execute upload_from_host / download_to_host copies bytes.
    Buffer* host_alias = nullptr;
  };

  // Allocate Buffers for the requested value_ids. out_buffers[i] is the
  // Buffer backing requests[i].value_id. Multiple requests sharing the
  // same mem_obj_id MAY yield Buffers that alias the same underlying
  // memory; that's the backend's choice.
  //
  // SYNCHRONOUS (init-time only).
  virtual ::executorch::runtime::Error allocate_all(
      ::executorch::runtime::Span<const AllocRequest> requests,
      ::executorch::runtime::Span<const ::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<Buffer*> out_buffers) = 0;

  // Materialize a graph constant on this runtime; persistent. The Instance
  // reads from the NamedDataMap directly so it can choose how to
  // materialize:
  //  - CPU: HostBuffer aliases the FreeableBuffer's region (zero-copy).
  //  - Apple-Silicon Metal: registers the FreeableBuffer's region with
  //    MetalStream; MetalBuffer aliases (zero-copy).
  //  - Discrete GPU: allocate device buffer + load_data_into directly
  //    (avoids host roundtrip).
  // The Instance owns the resulting Buffer AND the FreeableBuffer (so the
  // mmap'd region stays alive as long as the Buffer aliases it).
  // SYNCHRONOUS (init-time only). Constants do NOT go through
  // upload_from_host — they have their own dedicated path.
  virtual ::executorch::runtime::Result<Buffer*> upload_constant(
      const ::executorch::runtime::NamedDataMap& ndm,
      std::string_view key) = 0;

  // Provide the storage backing one Event slot. Called once per slot at
  // Plan construction.
  virtual std::unique_ptr<Event> make_event() = 0;

  //=== Async work issuance + sync helpers ===================================
  //
  // Public hot-path API: `execute` (intra-runtime kernel dispatch) and the
  // host↔device pair `upload_from_host` / `download_to_host` (cross-runtime
  // moves between CPU and a device runtime).
  //
  // The transfer methods only ever live on the **device** Instance:
  //   * upload_from_host:  CPU produced a value, this device consumes it.
  //   * download_to_host:  this device produced a value, CPU consumes it.
  // CpuInstance never has these called (CPU is always the host side); its
  // overrides return NotImplemented.
  //
  // We never need a runtime↔runtime path because v1 has at most one
  // non-host runtime. Any transfer between two non-host runtimes (future)
  // would route through host as two steps.

  // Make `dev_dst_buf` reflect `host_src_ptr`'s bytes by the time
  // `signal` reaches Complete. Implementation strategy is the runtime's
  // call:
  //
  //   - Host-addressable runtimes (CPU, Apple-Silicon Metal) typically
  //     re-alias `dev_dst_buf` to point at `host_src_ptr` (zero-copy).
  //     The "skip if dev_dst_buf->host_ptr() == host_src_ptr" check
  //     makes repeated executes with the same caller pointer free.
  //     If zero-copy alias fails (e.g. Metal refuses unaligned ptr),
  //     fall back to memcpy into the existing Buffer's storage.
  //   - Discrete GPU (Vulkan): real copy from host_src_ptr into
  //     pre-allocated VRAM via vkCmdCopyBuffer.
  //
  // Also propagates shape from src to dst's TensorImpl before signaling
  // (per the shape-on-event contract).
  //
  // Default: NotImplemented. Override in non-host Instances.
  virtual ::executorch::runtime::Error upload_from_host(
      ::executorch::runtime::EValue& /*host_src_ev*/,
      void* /*host_src_ptr*/,
      ::executorch::runtime::EValue& /*dev_dst_ev*/,
      Buffer* /*dev_dst_buf*/,
      QueueKind /*queue*/,
      ::executorch::runtime::Span<Event* const> /*wait_for*/,
      Event* /*signal*/) {
    return ::executorch::runtime::Error::NotImplemented;
  }

  // Symmetric: make `host_dst_ptr`'s bytes reflect `dev_src_buf`'s
  // contents. Same rebind-or-copy contract as upload_from_host.
  //
  // Default: NotImplemented. Override in non-host Instances.
  virtual ::executorch::runtime::Error download_to_host(
      ::executorch::runtime::EValue& /*dev_src_ev*/,
      Buffer* /*dev_src_buf*/,
      ::executorch::runtime::EValue& /*host_dst_ev*/,
      void* /*host_dst_ptr*/,
      QueueKind /*queue*/,
      ::executorch::runtime::Span<Event* const> /*wait_for*/,
      Event* /*signal*/) {
    return ::executorch::runtime::Error::NotImplemented;
  }

  // Inputs guaranteed on this runtime; outputs MUST end up here.
  // values: the LoadedDelegate's EValue array (carries dtype/shape and
  // scalar inputs). bindings: tensor storage backings (Buffer*) for
  // tensor value_ids. Both are needed by kernels: scalars come from
  // values[idx], tensor data comes from bindings.get(idx).
  //
  // SHAPE-ON-EVENT CONTRACT: by the time `signal` reaches
  // EventStatus::Complete, every output value's TensorImpl shape AND
  // bound Buffer bytes MUST be valid for downstream consumers.
  // Backends that determine output shapes synchronously inside execute()
  // (CPU portable kernels, Metal-with-MPSGraph metadata) update shape
  // before encoding the kernel. Backends whose output shapes are only
  // known after the GPU runs (e.g., Metal kernels that determine shape
  // post-execution) must register a completion handler that updates the
  // output TensorImpls' shape arrays AND THEN signals the event last.
  virtual ::executorch::runtime::Error execute(
      CompiledSegment* segment,
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      BindingView bindings,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) = 0;

  // Block CPU until event signals. Returns Error::Ok iff event reaches
  // EventStatus::Complete; returns the stored error if Failed/Poisoned.
  virtual ::executorch::runtime::Error wait(Event* event) = 0;

  // Stable per-Instance id within this Instance's RuntimeContext.
  virtual InstanceId id() const = 0;

  // Drain in-flight work *issued by this Instance only*. Blocks until
  // every currently outstanding submission this Instance has issued via
  // copy_*/execute reaches a terminal state. Idempotent.
  virtual void drain() = 0;

  // Forwards to the RuntimeContext's pool / recycler.
  virtual void release_buffer(Buffer* buf) = 0;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
