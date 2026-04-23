/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Instance.h>
#include <executorch/backends/portable/runtime_v2/metal/MetalEvent.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward decl: keep this header pure-C++; .mm includes MetalStream.h.
namespace executorch {
namespace backends {
namespace metal_v2 {
class MetalStream;
}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch

namespace executorch {
namespace backends {
namespace portable_v2 {

class MetalProvider;
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
  bool has_remap() const { return !value_remap_.empty(); }

 private:
  const ::executorch::backends::portable::Graph* graph_;
  std::vector<uint32_t> instruction_indices_;
  std::unordered_map<uint32_t, uint32_t> value_remap_;
};

/**
 * Metal Instance — dispatches ops via the existing metal_v2
 * MetalOpRegistry. Per-execute model:
 *   1. For each op in the segment: look up MetalOp; build EValue*
 *      vectors for inputs/outputs; sync TensorImpl::data_ptr to the
 *      bound MetalBuffer's host_ptr (= [mtlBuffer contents]); call
 *      op->dispatch(stream_, ins, outs).
 *   2. After all dispatches, stream_->flush() commits the live command
 *      buffer; install an addCompletedHandler that signals our event
 *      when the GPU finishes.
 *   3. On Apple Silicon (unified memory), bytes in the buffer are
 *      visible to host once the event signals.
 *
 * transfer_tensor handles cross-runtime hand-off via host pointers
 * (source's host_ptr → dst's host_ptr memcpy; works for CPU↔Metal
 * because both are HostBuffer-compatible on Apple Silicon).
 */
class MetalInstance final : public Instance {
 public:
  MetalInstance(MetalProvider* provider, InstanceId id);
  ~MetalInstance() override;

  ::executorch::runtime::Result<CompiledSegment*> compile_segment(
      const ::executorch::backends::portable::Graph& graph,
      ::executorch::runtime::Span<const uint32_t> instruction_indices,
      ::executorch::runtime::Span<const uint32_t> input_value_ids,
      ::executorch::runtime::Span<const uint32_t> output_value_ids,
      ::executorch::runtime::Span<const std::pair<uint32_t, uint32_t>>
          value_remap) override;

  ::executorch::runtime::Error allocate_all(
      ::executorch::runtime::Span<const AllocRequest> requests,
      ::executorch::runtime::Span<const ::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<Buffer*> out_buffers) override;

  ::executorch::runtime::Result<Buffer*> upload_constant(
      const ::executorch::runtime::NamedDataMap& ndm,
      std::string_view key) override;

  std::unique_ptr<Event> make_event() override;

  // Cross-runtime moves: only the device side overrides these.
  // Apple Silicon Metal: re-aliases the destination MetalBuffer to point
  // at the caller's host pointer (zero-copy via newBufferWithBytesNoCopy).
  // Falls back to memcpy into the original pool-allocated buffer if Metal
  // refuses zero-copy on the pointer.
  ::executorch::runtime::Error upload_from_host(
      ::executorch::runtime::EValue& host_src_ev,
      void* host_src_ptr,
      ::executorch::runtime::EValue& dev_dst_ev,
      Buffer* dev_dst_buf,
      QueueKind queue,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) override;

  ::executorch::runtime::Error download_to_host(
      ::executorch::runtime::EValue& dev_src_ev,
      Buffer* dev_src_buf,
      ::executorch::runtime::EValue& host_dst_ev,
      void* host_dst_ptr,
      QueueKind queue,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) override;

  ::executorch::runtime::Error execute(
      CompiledSegment* segment,
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      BindingView bindings,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) override;

  ::executorch::runtime::Error wait(Event* event) override;

  InstanceId id() const override { return id_; }

  void drain() override;

  void release_buffer(Buffer* buf) override;

 private:
  // Helper: drain wait_for events; if any is Failed/Poisoned, poison
  // signal and return AsyncDependencyFailed.
  ::executorch::runtime::Error check_dependencies_(
      ::executorch::runtime::Span<Event* const> wait_for, Event* signal);

  MetalProvider* provider_;  // not owned; provider outlives instance
  ::executorch::backends::metal_v2::MetalStream* stream_;  // borrowed from provider
  InstanceId id_;

  // Owns CompiledSegments returned from compile_segment.
  std::vector<std::unique_ptr<MetalCompiledSegment>> compiled_segments_;

  // Owns all MetalBuffers from allocate / upload_constant. I/O destination
  // Buffers may transition Owned → Aliasing in-place when upload_from_host
  // re-points them at caller storage; the destructor still works correctly
  // via mode_ tracking.
  std::vector<std::unique_ptr<MetalBuffer>> owned_buffers_;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
