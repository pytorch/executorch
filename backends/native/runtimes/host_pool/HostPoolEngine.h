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
#include <executorch/backends/native/runtimes/cpu/HostBuffer.h>
#include <executorch/backends/native/runtimes/host_pool/HostPoolRuntimeContext.h>

#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

/**
 * HostPoolEngine — the canonical home for boundary values (graph IO
 * and cross-runtime intermediates). Owns the host-resident HostBuffers;
 * compute providers (CPU, Metal, Vulkan, ...) get per-runtime
 * device-side mirrors.
 *
 * In the bid auction model, HostPool is the fallback claimant: it
 * claims every HostMirror / HostOnly request not previously claimed by
 * a device engine (typical case: a CUDA engine offering pinned host
 * memory; otherwise HostPool wins). HostPool declines DeviceMirror /
 * DeviceOnly requests.
 *
 * Does not run kernels. compile_segment / execute return NotSupported.
 */
class HostPoolEngine final : public Engine {
 public:
  explicit HostPoolEngine(HostPoolRuntimeContext& ctx, InstanceId id)
      : ctx_(ctx), id_(id) {}

  ~HostPoolEngine() override = default;

  ::executorch::runtime::Result<CompiledSegment*> compile_segment(
      const ::executorch::backends::portable::Graph& /*graph*/,
      ::executorch::runtime::Span<const uint32_t> /*instruction_indices*/,
      ::executorch::runtime::Span<const uint32_t> /*input_value_ids*/,
      ::executorch::runtime::Span<const uint32_t> /*output_value_ids*/,
      ::executorch::runtime::Span<const std::pair<uint32_t, uint32_t>>
          /*value_remap*/) override {
    return ::executorch::runtime::Error::NotSupported;
  }

  ::executorch::runtime::Error allocate_buffers(
      ::executorch::runtime::Span<const AllocRequest> requests,
      ::executorch::runtime::Span<::executorch::runtime::EValue> values,
      ::executorch::runtime::Span<AllocClaim> out_claims) override;

  ::executorch::runtime::Error upload_constants(
      const ::executorch::runtime::NamedDataMap& /*ndm*/,
      ::executorch::runtime::Span<const ConstRequest> /*requests*/) override {
    return ::executorch::runtime::Error::NotSupported;
  }

  std::unique_ptr<Event> make_event() override {
    return std::make_unique<CpuEvent>();
  }

  // Per-execute IO binding. Re-aliases the host HostBuffer for each
  // value_id to caller's storage; sets data_ptr on the central EValue.
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

  // Host re-alias path: read source data_ptr from EValue; copy or
  // skip-if-same into the host destination Buffer's storage.
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
      CompiledSegment* /*segment*/,
      ::executorch::runtime::Span<::executorch::runtime::EValue> /*values*/,
      ::executorch::runtime::Span<Event* const> /*wait_for*/,
      Event* /*signal*/) override {
    return ::executorch::runtime::Error::NotSupported;
  }

  ::executorch::runtime::Error wait(Event* /*event*/) override {
    return ::executorch::runtime::Error::Ok;
  }

  InstanceId id() const override { return id_; }

  void drain() override {}

 private:
  HostPoolRuntimeContext& ctx_;
  InstanceId id_;

  // Single contiguous arena that backs every Owned (non-IO) Buffer
  // returned by allocate_buffers. Buffer data_ptrs are offsets into
  // this chunk. Allocated once at allocate_buffers; freed at
  // ~HostPoolEngine.
  struct AlignedFree {
    void operator()(uint8_t* p) const noexcept { std::free(p); }
  };
  std::unique_ptr<uint8_t[], AlignedFree> arena_;
  size_t arena_size_ = 0;

  std::vector<std::unique_ptr<HostBuffer>> owned_buffers_;

  // Per-engine value_id → Buffer table. Indexes EVERY HostBuffer this
  // engine owns (intermediates, IO-bound). Populated by
  // allocate_buffers, bind_inputs, bind_outputs.
  std::unordered_map<uint32_t, HostBuffer*> value_to_buffer_;

  // Helper: re-alias (or allocate fresh Aliasing) the HostBuffer for
  // value_id to caller_ptr; sets data_ptr on the central EValue.
  ::executorch::runtime::Error bind_one_(
      ::executorch::runtime::EValue& central_ev,
      uint32_t value_id,
      const ::executorch::runtime::EValue& caller_ev);
};

}  // namespace native
}  // namespace backends
}  // namespace executorch
