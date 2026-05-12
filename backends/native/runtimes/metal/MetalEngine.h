/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Engine.h>
#include <executorch/backends/native/core/EngineUtils.h>
#include <executorch/backends/native/runtimes/metal/MetalEvent.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Forward decl: keep this header pure-C++; .mm includes MetalStream.h.
namespace executorch {
namespace backends {
namespace metal_v2 {
class MetalStream;
} // namespace metal_v2
} // namespace backends
} // namespace executorch

namespace executorch {
namespace backends {
namespace native {

class MetalRuntime;
class MetalBuffer;

/**
 * CompiledSegment for Metal: stores instruction indices and the
 * value_remap (analogous to CpuCompiledSegment). Kernel compilation
 * happens lazily inside MetalOp::getKernel during dispatch.
 */
class MetalCompiledSegment final : public CompiledSegment {
 public:
  MetalCompiledSegment(
      const ::executorch::backends::portable::Graph* graph,
      std::vector<uint32_t> instruction_indices,
      std::unordered_map<uint32_t, uint32_t> value_remap)
      : graph_(graph),
        instruction_indices_(std::move(instruction_indices)),
        value_remap_(std::move(value_remap)) {}

  const ::executorch::backends::portable::Graph* graph() const {
    return graph_;
  }
  const std::vector<uint32_t>& instruction_indices() const {
    return instruction_indices_;
  }
  uint32_t remap(uint32_t v) const {
    auto it = value_remap_.find(v);
    return it != value_remap_.end() ? it->second : v;
  }
  bool has_remap() const {
    return !value_remap_.empty();
  }

 private:
  const ::executorch::backends::portable::Graph* graph_;
  std::vector<uint32_t> instruction_indices_;
  std::unordered_map<uint32_t, uint32_t> value_remap_;
};

/**
 * Metal Engine — dispatches ops via the existing metal_v2
 * MetalOpRegistry. Owns the MetalBuffers it allocated; releases at
 * destruction.
 */
class MetalEngine final : public DeviceEngine {
 public:
  MetalEngine(
      const ::executorch::backends::portable::Graph& graph,
      MetalRuntime* provider,
      InstanceId id);
  ~MetalEngine() override;

  ::executorch::runtime::Result<CompiledSegment*> compile_segment(
      ::executorch::runtime::Span<const uint32_t> instruction_indices,
      ::executorch::runtime::Span<const uint32_t> input_value_ids,
      ::executorch::runtime::Span<const uint32_t> output_value_ids,
      ::executorch::runtime::Span<const std::pair<uint32_t, uint32_t>>
          value_remap) override;

  ::executorch::runtime::Error allocate_buffers(
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<const AllocRequest> requests,
      ::executorch::runtime::Span<AllocClaim> out_claims) override;

  // Dynamic-shape resize: lazily allocate (or grow) the MetalBuffer
  // for value_id. Reuses existing buffer if it has enough capacity;
  // otherwise allocates a fresh pool buffer of the new size.
  ::executorch::runtime::Error resize_tensor(
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      uint32_t value_id,
      ::executorch::runtime::ArrayRef<::executorch::aten::SizesType> new_sizes)
      override;

  ::executorch::runtime::Error upload_constants(
      const ::executorch::runtime::NamedDataMap* ndm,
      ::executorch::runtime::Span<const ConstRequest> requests) override;

  std::unique_ptr<Event> make_event() const override;

  // Per-execute IO binding. Re-aliases the engine's MetalBuffer for
  // each owned mirror_id to caller's storage. Tries zero-copy via
  // registerExternalBuffer; falls back to memcpy into pool storage on
  // alignment refusal. Self-filtered via internal io_*_bindings_.
  ::executorch::runtime::Error bind_inputs(
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<const ::executorch::runtime::EValue* const>
input_args) override;

  ::executorch::runtime::Error bind_outputs(
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<::executorch::runtime::EValue* const>
          output_args) override;

  // Build internal IO bindings: walk graph IO; for each (idx, vid)
  // that has a DeviceMirror this engine claimed (host_to_mirror_id_),
  // record (idx, mirror_id). Direct-handled IO (no mirror) records
  // (idx, vid) so bind_one_ can lazily wrap the caller's pointer.
  ::executorch::runtime::Error set_io_bindings(
      ::executorch::runtime::Span<const InputBinding> graph_inputs,
      ::executorch::runtime::Span<const OutputBinding> graph_outputs) override;

  // Direct-handle graph INPUTS: on Apple-Silicon UMA we wrap the
  // caller's host pointer in a fresh aliasing MetalBuffer per execute.
  // Skips the router-side DeviceMirror allocation + per-execute
  // input TransferStep.
  //
  // Inputs only — NOT outputs. Reason: at bind_outputs time we only
  // see the caller's current output Tensor sizes, but the kernel
  // determines the actual output bytes during compute. With dynamic
  // shapes (B=1 → B=5), bind sees 48 bytes from the prior call's
  // resize_tensor while the kernel needs 80 — bufferForPtr asserts.
  // For outputs, the mirror+TransferStep path remains: router
  // pre-allocates at tensor_nbytes_max so the kernel always fits.
  //
  // Dynamic shapes: bind_one_ tracks all our registerExternalBuffer
  // calls in our_registrations_ and uses the new
  // MetalAllocator::unregisterExternalBuffer to cleanly drop the prior
  // entry before re-registering at a new size or ptr.
  bool handles_input_directly(uint32_t /*vid*/) const override {
    return true;
  }
  bool handles_output_directly(uint32_t /*vid*/) const override {
    return false;
  }

  // Cross-runtime moves: only the device side overrides these.
  ::executorch::runtime::Error upload_from_host(
      const ::executorch::runtime::EValue& host_src_ev,
      ::executorch::runtime::EValue& dev_dst_ev,
      uint32_t dev_dst_value_id,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) override;

  ::executorch::runtime::Error download_to_host(
      const ::executorch::runtime::EValue& dev_src_ev,
      uint32_t dev_src_value_id,
      ::executorch::runtime::EValue& host_dst_ev,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) override;

  ::executorch::runtime::Error execute(
      const CompiledSegment* segment,
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) override;

  ::executorch::runtime::Error wait(Event* event) const override;

  InstanceId id() const override {
    return id_;
  }

  void drain() override;

 private:
  // Helper: drain wait_for events; if any is Failed/Poisoned, poison
  // signal and return AsyncDependencyFailed.
  ::executorch::runtime::Error check_dependencies_(
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal);

  // Helper: re-alias (or allocate fresh Aliasing) the MetalBuffer for
  // value_id to caller_ptr; sets data_ptr on the central EValue.
  ::executorch::runtime::Error bind_one_(
      ::executorch::runtime::EValue& central_ev,
      uint32_t value_id,
      const ::executorch::runtime::EValue& caller_ev);

  MetalRuntime* provider_; // not owned; provider outlives instance
  ::executorch::backends::metal_v2::MetalStream*
      stream_; // borrowed from provider
  InstanceId id_;

  // Owns CompiledSegments returned from compile_segment.
  std::vector<std::unique_ptr<MetalCompiledSegment>> compiled_segments_;

  // Owns all MetalBuffers from allocate_buffers / upload_constants /
  // bind_inputs / bind_outputs. Released at ~MetalEngine.
  std::vector<std::unique_ptr<MetalBuffer>> owned_buffers_;

  // Owns TensorImpl + sizes/dim_order/strides storage for any
  // router-minted mirror value_ids this engine claims (DeviceMirror
  // requests). Materialized in allocate_buffers.
  std::vector<TensorImplStorage> mirror_tensor_metas_;

  // Per-engine value_id → Buffer table.
  std::unordered_map<uint32_t, MetalBuffer*> value_to_buffer_;

  // For DeviceMirror requests: maps host_mirror_value_id -> mirror_id
  // (= the engine's internal vid). Populated during allocate_buffers.
  // Used by set_io_bindings to find which mirror_id to bind for each
  // graph IO vid.
  std::unordered_map<uint32_t, uint32_t> host_to_mirror_id_;

  // Internal IO bindings populated by set_io_bindings. (graph_io_idx ->
  // mirror_id this engine binds for that slot.)
  struct IoBinding {
    uint32_t graph_idx;
    uint32_t internal_vid;
  };
  // Set of host pointers we currently hold registerExternalBuffer
  // entries for. Includes both per-execute IO direct-handling binds
  // (via bind_one_) and engine-lifetime constant uploads (via
  // upload_constants — those use ResidencyClass::Permanent). bind_one_
  // mutates this set during normal operation (drop+re-register on
  // size grow / ptr change / cross-vid collision); upload_constants
  // is insert-only. ~MetalEngine walks the set and calls
  // unregisterExternalBuffer(ptr) for each — this is what avoids
  // MetalAllocator's "Permanent External entries the caller forgot
  // to unregister" leak warning at process shutdown.
  std::unordered_set<void*> our_registrations_;
  std::vector<IoBinding> io_input_bindings_;
  std::vector<IoBinding> io_output_bindings_;
};

} // namespace native
} // namespace backends
} // namespace executorch
