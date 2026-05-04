/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Engine.h>
#include <executorch/backends/native/runtimes/metal/MetalEvent.h>

#include <memory>
#include <unordered_map>
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
class MetalEngine final : public Engine {
 public:
  MetalEngine(MetalRuntime* provider, InstanceId id);
  ~MetalEngine() override;

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

  // Per-execute IO binding: re-alias the engine's MetalBuffer for each
  // value_id to caller's storage. Tries zero-copy via
  // registerExternalBuffer; falls back to memcpy into pool storage on
  // alignment refusal.
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

  // Cross-runtime moves: only the device side overrides these.
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

  // Per-engine value_id → Buffer table.
  std::unordered_map<uint32_t, MetalBuffer*> value_to_buffer_;
};

} // namespace native
} // namespace backends
} // namespace executorch
