/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Buffer.h>
#include <executorch/backends/native/core/Engine.h>
#include <executorch/backends/native/core/Event.h>
#include <executorch/backends/native/ir/GraphTypes.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

/**
 * Helpers shared by all backend Engine implementations.
 *
 * These deliberately operate against the public Event interface (no
 * dynamic_cast to backend-specific subclasses).
 */

// Block on each event in `wait_for` until it reaches a terminal state.
// If any reached Failed / Poisoned, poison `signal` with the upstream
// error and return InternalError. Otherwise return Ok.
//
// Backends call this at the top of execute / upload_from_host /
// download_to_host. Sync producers settle before this runs (fast
// path); async producers cause this to block via Event::wait_until_settled
// (CpuEvent uses bounded spin + condvar).
inline ::executorch::runtime::Error check_async_dependencies(
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  for (Event* dep : wait_for) {
    if (!dep)
      continue;
    dep->wait_until_settled();
    EventStatus s = dep->status();
    if (s == EventStatus::Failed || s == EventStatus::Poisoned) {
      if (signal)
        signal->signal_poisoned(dep->error());
      return ::executorch::runtime::Error::Internal;
    }
  }
  return ::executorch::runtime::Error::Ok;
}

/**
 * RAII guard ensuring a signal Event reaches a terminal state by the time
 * the guarded scope exits.
 *
 * Usage:
 *   Error MyInstance::execute(..., Event* signal) {
 *     SignalGuard guard(signal);
 *     if (auto e = check_async_dependencies(wait_for, signal); e != Ok)
 *       return e;                       // signal already Poisoned
 *     if (bad_input) {
 *       guard.fail(InvalidArgument);    // explicit failure
 *       return InvalidArgument;
 *     }
 *     // ... do work ...
 *     guard.complete();                 // success
 *     return Ok;
 *   }
 *
 * If the function returns without calling complete()/fail(), the guard's
 * destructor inspects the signal's status: if still Pending, it fires
 * signal_failed(Internal). Failed/Poisoned/Complete states are left
 * untouched (sticky).
 */
class SignalGuard {
 public:
  explicit SignalGuard(Event* signal) noexcept : signal_(signal) {}
  ~SignalGuard() {
    if (signal_ && signal_->status() == EventStatus::Pending) {
      signal_->signal_failed(::executorch::runtime::Error::Internal);
    }
  }
  SignalGuard(const SignalGuard&) = delete;
  SignalGuard& operator=(const SignalGuard&) = delete;

  // Settle the signal as Complete. Idempotent; subsequent calls no-op.
  void complete() {
    if (signal_) {
      signal_->prepare_signal();
      signal_->signal_complete();
      signal_ = nullptr;
    }
  }

  // Settle the signal as Failed with the given error. Idempotent.
  void fail(::executorch::runtime::Error e) {
    if (signal_) {
      signal_->signal_failed(e);
      signal_ = nullptr;
    }
  }

 private:
  Event* signal_;
};

/**
 * For an Engine::allocate_buffers() request list, compute the MAX nbytes
 * across all requests sharing each non-negative mem_obj_id. Backends
 * use this so a single Buffer per mem_obj_id group is sized to fit the
 * largest tensor any sharer may hold (capacity, not the per-request
 * logical size). Per-execute logical-size changes happen via
 * resize_tensor on TensorImpls and never require physical reallocation.
 *
 * Requests with mem_obj_id < 0 (dedicated allocations) are not
 * included in the result map; callers fall back to the request's own
 * tensor.nbytes() for those.
 *
 * Caller must validate that every request's value_id is in range and
 * isTensor() before invoking; this helper assumes well-formed input
 * and reads the EValue tensor unchecked.
 */
inline std::unordered_map<int32_t, size_t> compute_mem_obj_capacity(
    ::executorch::runtime::Span<const Engine::AllocRequest> requests,
    ::executorch::runtime::Span<const ::executorch::runtime::EValue> values) {
  std::unordered_map<int32_t, size_t> caps;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.role != BufferRole::Internal)
      continue;
    if (req.value_id >= values.size() || !values[req.value_id].isTensor()) {
      continue; // caller will surface the error in its own validation pass
    }
    size_t nb = values[req.value_id].toTensor().nbytes();
    auto& cur = caps[req.mem_obj_id];
    if (nb > cur)
      cur = nb;
  }
  return caps;
}

/**
 * Plan an arena layout for a batch of AllocRequests on a host-addressable
 * backend.
 *
 * Three buckets (per request):
 *   1. mem_obj_id < 0 → deferred (offsets[i] = SIZE_MAX, sizes[i] = 0).
 *      Caller should not place these in arena.
 *   2. partner_host_ptr non-null (DeviceMirror with host-addressable
 *      partner already known via values[host_mirror_value_id].data_ptr)
 *      → backend wraps the partner's pointer directly. Doesn't take an
 *      arena slot. offsets[i] = SIZE_MAX, sizes[i] = 0.
 *   3. Otherwise → arena slot, grouped by mem_obj_id. Each group's slot
 *      is sized to MAX nbytes across the GROUP'S NON-ALIASED members
 *      on this Engine.
 *
 * `partner_host_ptr_for(host_vid) -> void*` is a callback the caller
 * supplies to look up the partner host_ptr for a DeviceMirror request's
 * host_mirror_value_id. Callers typically read
 * values[host_mirror_value_id].toTensor().mutable_data_ptr().
 *
 * Returns offsets[i], sizes[i] for arena-allocated requests; the
 * backend constructs Aliasing Buffers at base + offsets[i].
 */
struct ArenaLayout {
  size_t total_bytes = 0;
  std::vector<size_t> offsets; // SIZE_MAX = no arena slot for this request
  std::vector<size_t> sizes;
  std::unordered_map<int32_t, size_t> group_offsets;
};

template <typename PartnerLookup>
inline ArenaLayout compute_arena_layout(
    ::executorch::runtime::Span<const Engine::AllocRequest> requests,
    ::executorch::runtime::Span<const ::executorch::runtime::EValue> values,
    size_t alignment,
    PartnerLookup partner_host_ptr_for) {
  auto align_up = [alignment](size_t v) {
    return (v + alignment - 1) & ~(alignment - 1);
  };
  ArenaLayout layout;
  layout.offsets.assign(requests.size(), SIZE_MAX);
  layout.sizes.assign(requests.size(), 0);

  // Compute group capacities considering ONLY non-aliased,
  // non-deferred requests. Aliased requests will wrap their own
  // partner's storage; they don't need to fit in the arena slot.
  std::unordered_map<int32_t, size_t> group_caps;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.role != BufferRole::Internal) continue;
    if (req.host_mirror_value_id != Engine::kInvalidValueId &&
        partner_host_ptr_for(req.host_mirror_value_id) != nullptr) {
      continue;
    }
    if (req.value_id >= values.size() ||
        !values[req.value_id].isTensor()) continue;
    size_t nb = values[req.value_id].toTensor().nbytes();
    auto& cur = group_caps[req.mem_obj_id];
    if (nb > cur) cur = nb;
  }

  // Lay out arena slots, one per non-aliased, non-deferred group.
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.role != BufferRole::Internal) continue;
    if (req.host_mirror_value_id != Engine::kInvalidValueId &&
        partner_host_ptr_for(req.host_mirror_value_id) != nullptr) {
      continue;
    }
    auto cit = group_caps.find(req.mem_obj_id);
    if (cit == group_caps.end()) continue;
    auto it = layout.group_offsets.find(req.mem_obj_id);
    if (it == layout.group_offsets.end()) {
      size_t off = align_up(layout.total_bytes);
      layout.group_offsets[req.mem_obj_id] = off;
      layout.offsets[i] = off;
      layout.sizes[i] = cit->second;
      layout.total_bytes = off + cit->second;
    } else {
      layout.offsets[i] = it->second;
      layout.sizes[i] = cit->second;
    }
  }
  return layout;
}

} // namespace native
} // namespace backends
} // namespace executorch
