/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Instance.h>
#include <executorch/backends/portable/runtime_v2/cpu/CpuEvent.h>
#include <executorch/backends/portable/runtime_v2/cpu/CpuRuntimeContext.h>
#include <executorch/backends/portable/runtime_v2/cpu/HostBuffer.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * CompiledSegment for CPU: just remembers which instructions to run.
 * No actual compilation — CPU dispatches via the portable kernel
 * registry per-instruction at execute time.
 *
 * value_remap rewrites graph value_ids to the value_ids the segment
 * should consult in the BindingTable. The router uses this when it
 * synthesizes new value_ids for cross-runtime transfer destinations.
 */
class CpuCompiledSegment final : public CompiledSegment {
 public:
  CpuCompiledSegment(
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
 * CPU Instance — synchronous execution via the existing portable kernel
 * registry. All work-issuing methods complete before returning; events
 * transition to Complete or Failed in-place.
 */
class CpuInstance final : public Instance {
 public:
  explicit CpuInstance(CpuRuntimeContext& ctx, InstanceId id)
      : ctx_(ctx), id_(id) {}

  ~CpuInstance() override;

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

  // CpuInstance overrides these to re-alias the destination HostBuffer
  // to point at the caller's host pointer (zero-copy). Used for graph
  // I/O bindings on the CPU side and for cross-runtime intermediates
  // where CPU is the consumer.
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

  void drain() override {}  // CPU is synchronous; no in-flight work.

  void release_buffer(Buffer* buf) override;

 private:
  CpuRuntimeContext& ctx_;
  InstanceId id_;

  // Owns all Buffers allocated via allocate() / upload_constant().
  // I/O destination Buffers may transition from Owned → Aliasing
  // in-place via HostBuffer::re_alias when upload_from_host re-points
  // them at caller storage; the destructor still works correctly via
  // mode_ tracking.
  std::vector<std::unique_ptr<HostBuffer>> owned_buffers_;

  // Owns CompiledSegments returned from compile_segment.
  std::vector<std::unique_ptr<CpuCompiledSegment>> compiled_segments_;

  // Helper: check wait_for for poison; if any, signal poisons signal and
  // returns AsyncDependencyFailed.
  ::executorch::runtime::Error check_dependencies_(
      ::executorch::runtime::Span<Event* const> wait_for, Event* signal);
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
