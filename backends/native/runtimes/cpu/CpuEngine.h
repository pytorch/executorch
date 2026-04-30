/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Engine.h>
#include <executorch/backends/native/runtimes/cpu/CpuEvent.h>
#include <executorch/backends/native/runtimes/cpu/CpuRuntimeContext.h>
#include <executorch/backends/native/runtimes/cpu/HostBuffer.h>

#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

/**
 * CompiledSegment for CPU: just remembers which instructions to run.
 * No actual compilation — CPU dispatches via the portable kernel
 * registry per-instruction at execute time.
 *
 * value_remap rewrites graph value_ids to the value_ids the segment
 * should consult in the engine's value→Buffer table. The router uses
 * this when it mints new value_ids for cross-runtime mirror destinations.
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
  bool has_remap() const {
    return !value_remap_.empty();
  }

 private:
  const ::executorch::backends::portable::Graph* graph_;
  std::vector<uint32_t> instruction_indices_;
  std::unordered_map<uint32_t, uint32_t> value_remap_;
};

/**
 * CPU Engine — synchronous execution via the existing portable kernel
 * registry. All work-issuing methods complete before returning; events
 * transition to Complete or Failed in-place.
 */
class CpuEngine final : public Engine {
 public:
  explicit CpuEngine(CpuRuntimeContext& ctx, InstanceId id)
      : ctx_(ctx), id_(id) {}

  ~CpuEngine() override;

  ::executorch::runtime::Result<CompiledSegment*> compile_segment(
      const ::executorch::backends::portable::Graph& graph,
      ::executorch::runtime::Span<const uint32_t> instruction_indices,
      ::executorch::runtime::Span<const uint32_t> input_value_ids,
      ::executorch::runtime::Span<const uint32_t> output_value_ids,
      ::executorch::runtime::Span<const std::pair<uint32_t, uint32_t>>
          value_remap) override;

  ::executorch::runtime::Error allocate_buffers(
      ::executorch::runtime::Span<const AllocRequest> requests,
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<AllocClaim> out_claims) override;

  ::executorch::runtime::Error upload_constants(
      const ::executorch::runtime::NamedDataMap& ndm,
      ::executorch::runtime::Span<const ConstRequest> requests) override;

  std::unique_ptr<Event> make_event() override;

  // Per-execute IO binding: re-alias the engine's HostBuffer for each
  // value_id to caller's storage. For the first call, allocates a
  // fresh Aliasing HostBuffer; subsequent calls re-target in place.
  ::executorch::runtime::Error bind_inputs(
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<const ::executorch::runtime::EValue>
          caller_evs,
      ::executorch::runtime::Span<const uint32_t> value_ids) override;

  ::executorch::runtime::Error bind_outputs(
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<const ::executorch::runtime::EValue>
          caller_evs,
      ::executorch::runtime::Span<const uint32_t> value_ids) override;

  // CPU is host hardware: it can read/write any host pointer directly.
  // Graph IO already has a HostMirror/HostOnly Buffer on the host pool
  // that bind_inputs/bind_outputs re-aliases to the caller's pointer
  // each execute. CPU's compiled segment references the original IO
  // vid (no remap), and CpuEngine::execute's refresh() silently no-ops
  // when the vid isn't in value_to_buffer_, preserving the caller's
  // data_ptr that the host pool wrote onto values[vid].
  //
  // Opting out avoids a redundant DeviceMirror alloc and a no-op IO
  // TransferStep per execute.
  bool wants_input_mirror(uint32_t /*vid*/) const override { return false; }
  bool wants_output_mirror(uint32_t /*vid*/) const override { return false; }

  // CpuEngine overrides these so cross-runtime transfer steps where
  // CPU is the device side can resolve internally. Bytes flow via host
  // pointers (the value's storage on this engine is host-addressable).
  ::executorch::runtime::Error upload_from_host(
      ::executorch::runtime::EValue& host_src_ev,
      ::executorch::runtime::EValue& dev_dst_ev,
      uint32_t dev_dst_value_id,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) override;

  ::executorch::runtime::Error download_to_host(
      ::executorch::runtime::EValue& dev_src_ev,
      uint32_t dev_src_value_id,
      ::executorch::runtime::EValue& host_dst_ev,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) override;

  ::executorch::runtime::Error execute(
      CompiledSegment* segment,
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal) override;

  ::executorch::runtime::Error wait(Event* event) override;

  InstanceId id() const override {
    return id_;
  }

  void drain() override {} // CPU is synchronous; no in-flight work.

 private:
  CpuRuntimeContext& ctx_;
  InstanceId id_;

  // Single contiguous arena that backs every Owned (non-IO,
  // non-mirror) Buffer returned by allocate_buffers. Each Buffer's
  // data_ptr is an offset into this chunk. Allocated once at
  // allocate_buffers; freed at ~CpuEngine. Declared BEFORE
  // owned_buffers_ so destruction order frees Aliasing HostBuffers
  // first, then the arena.
  struct AlignedFree {
    void operator()(uint8_t* p) const noexcept { std::free(p); }
  };
  std::unique_ptr<uint8_t[], AlignedFree> arena_;
  size_t arena_size_ = 0;

  // Owns all Buffers allocated via allocate_buffers / upload_constants /
  // bind_inputs / bind_outputs. Released at ~CpuEngine.
  std::vector<std::unique_ptr<HostBuffer>> owned_buffers_;

  // Per-engine value_id → Buffer table. Indexes EVERY Buffer this
  // engine owns (intermediates, mirrors, constants, IO-bound).
  // Populated by allocate_buffers, upload_constants, bind_inputs,
  // bind_outputs. Consumed by execute (kernel-arg lookup) and
  // upload_from_host / download_to_host.
  std::unordered_map<uint32_t, HostBuffer*> value_to_buffer_;

  // Owns CompiledSegments returned from compile_segment.
  std::vector<std::unique_ptr<CpuCompiledSegment>> compiled_segments_;

  // Helper: check wait_for for poison; if any, signal poisons signal and
  // returns AsyncDependencyFailed.
  ::executorch::runtime::Error check_dependencies_(
      ::executorch::runtime::Span<Event* const> wait_for,
      Event* signal);

  // Helper: re-alias (or allocate fresh Aliasing) the HostBuffer for
  // value_id to caller_ptr; sets data_ptr on the central EValue.
  ::executorch::runtime::Error bind_one_(
      ::executorch::runtime::EValue& central_ev,
      uint32_t value_id,
      const ::executorch::runtime::EValue& caller_ev);
};

} // namespace native
} // namespace backends
} // namespace executorch
