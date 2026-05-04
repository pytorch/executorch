/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/runtimes/host_pool/HostPoolEngine.h>

#include <executorch/backends/native/core/EngineUtils.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace native {

::executorch::runtime::Error HostPoolEngine::allocate_buffers(
    ::executorch::runtime::Span<const AllocRequest> requests,
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<AllocClaim> out_claims) {
  if (requests.size() != out_claims.size()) {
    return ::executorch::runtime::Error::InvalidArgument;
  }
  // Sanity-check inputs first.
  for (const auto& req : requests) {
    if (req.value_id >= values.size() || !values[req.value_id].isTensor()) {
      ET_LOG(Error,
             "HostPoolEngine::allocate_buffers: value_id=%u missing or not a tensor",
             req.value_id);
      return ::executorch::runtime::Error::InvalidArgument;
    }
  }

  // HostPool is the floor: claims every HostMirror / HostOnly request
  // not previously claimed by a device engine. Declines DeviceMirror /
  // DeviceOnly (those are device-engine territory).
  //
  // "Previously claimed" detection: we check
  //   values[req.value_id].toTensor().data_ptr()
  // — a non-null data_ptr means a device engine already wrote a host
  // pointer for this vid (e.g., MetalEngine's UMA collapse), so HostPool
  // skips this request and marks it Claimed (the device engine owns it).

  // First pass: mark claim status, identify which requests need arena.
  std::vector<bool> needs_arena(requests.size(), false);
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.kind == MemoryKind::DeviceMirror ||
        req.kind == MemoryKind::DeviceOnly) {
      out_claims[i] = AllocClaim::Declined;
      continue;
    }
    // Check if a device engine already claimed (via populated data_ptr).
    void* existing =
        values[req.value_id].toTensor().mutable_data_ptr();
    if (existing != nullptr) {
      // Already claimed by an earlier engine (UMA collapse path).
      // Mark as Claimed so NativeBackend's bookkeeping is satisfied,
      // but don't allocate ourselves.
      out_claims[i] = AllocClaim::Claimed;
      continue;
    }
    out_claims[i] = AllocClaim::Claimed;
    if (req.role != BufferRole::Internal) {
      // Graph IO: claim ownership but defer storage to per-execute
      // bind_inputs / bind_outputs, which allocates an Aliasing
      // HostBuffer wrapping caller's pointer. No arena slot needed.
      ET_LOG(Debug,
             "[mem] host_pool: value_id=%u kind=%s role=%s (deferred to bind_io)",
             req.value_id, to_string(req.kind), to_string(req.role));
      continue;
    }
    needs_arena[i] = true;
  }

  // Compute arena layout for non-skipped, non-deferred requests.
  // mem_obj_id < 0 → deferred (offsets[i] = SIZE_MAX). Today's router
  // doesn't emit such for non-IO host allocations (graph IO bypasses
  // allocate_buffers entirely).
  constexpr size_t kAlignment = 16;
  auto align_up = [](size_t v) {
    return (v + kAlignment - 1) & ~(kAlignment - 1);
  };

  // Per-mem_obj_id capacity over needs_arena requests.
  std::unordered_map<int32_t, size_t> group_caps;
  for (size_t i = 0; i < requests.size(); ++i) {
    if (!needs_arena[i]) continue;
    const auto& req = requests[i];
    size_t nb = values[req.value_id].toTensor().nbytes();
    auto& cur = group_caps[req.mem_obj_id];
    if (nb > cur) cur = nb;
  }

  // Lay out arena slots; one per (non-deferred) group.
  std::unordered_map<int32_t, size_t> group_offsets;
  size_t arena_total = 0;
  for (size_t i = 0; i < requests.size(); ++i) {
    if (!needs_arena[i]) continue;
    const auto& req = requests[i];
    if (req.mem_obj_id < 0) continue; // deferred
    if (group_offsets.count(req.mem_obj_id)) continue;
    size_t off = align_up(arena_total);
    group_offsets[req.mem_obj_id] = off;
    arena_total = off + group_caps[req.mem_obj_id];
  }

  if (arena_total > 0) {
    size_t alloc_bytes = align_up(arena_total);
    arena_.reset(
        static_cast<uint8_t*>(std::aligned_alloc(kAlignment, alloc_bytes)));
    if (!arena_) return ::executorch::runtime::Error::MemoryAllocationFailed;
    arena_size_ = alloc_bytes;
    ET_LOG(Debug,
           "[mem] host_pool: arena allocated %zu bytes at %p",
           arena_size_, arena_.get());
  }

  // Hand out Aliasing HostBuffers. Dedup per mem_obj_id.
  std::unordered_map<int32_t, HostBuffer*> mem_obj_buffers;
  for (size_t i = 0; i < requests.size(); ++i) {
    if (!needs_arena[i]) continue;
    const auto& req = requests[i];
    if (req.mem_obj_id < 0) continue; // deferred (no arena slot)

    HostBuffer* hb_raw = nullptr;
    auto it = mem_obj_buffers.find(req.mem_obj_id);
    if (it != mem_obj_buffers.end()) {
      hb_raw = it->second;
    } else {
      size_t cap = group_caps[req.mem_obj_id];
      size_t off = group_offsets[req.mem_obj_id];
      std::unique_ptr<HostBuffer> hb(
          HostBuffer::alias(arena_.get() + off, cap, req.kind));
      if (!hb) return ::executorch::runtime::Error::MemoryAllocationFailed;
      hb_raw = hb.get();
      mem_obj_buffers[req.mem_obj_id] = hb_raw;
      owned_buffers_.push_back(std::move(hb));
    }
    value_to_buffer_[req.value_id] = hb_raw;
    // Host-addressable: write data_ptr into central EValue so other
    // engines (DeviceMirror partners) and the executor see it.
    values[req.value_id]
        .toTensor()
        .unsafeGetTensorImpl()
        ->set_data(hb_raw->host_ptr());
    ET_LOG(Debug,
           "[mem] host_pool: value_id=%u kind=%s mem_obj_id=%d host_ptr=%p",
           req.value_id, to_string(req.kind), req.mem_obj_id, hb_raw->host_ptr());
  }
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error HostPoolEngine::bind_one_(
    ::executorch::runtime::EValue& central_ev,
    uint32_t value_id,
    const ::executorch::runtime::EValue& caller_ev) {
  if (!caller_ev.isTensor() || !central_ev.isTensor()) {
    return ::executorch::runtime::Error::InvalidArgument;
  }
  auto& caller_t = caller_ev.toTensor();
  void* host_ptr = caller_t.mutable_data_ptr();
  size_t nbytes = caller_t.nbytes();
  if (!host_ptr) return ::executorch::runtime::Error::InvalidArgument;

  // Resize central TensorImpl to caller's actual shape.
  auto& central_t = central_ev.toTensor();
  if (auto e = ::executorch::runtime::resize_tensor(central_t, caller_t.sizes());
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }

  auto it = value_to_buffer_.find(value_id);
  if (it != value_to_buffer_.end()) {
    HostBuffer* hb = it->second;
    hb->re_alias(host_ptr, nbytes);
    central_t.unsafeGetTensorImpl()->set_data(host_ptr);
    ET_LOG(Debug,
           "[mem] host_pool: bind dst=%u host_ptr=%p bytes=%zu (re-aliased)",
           value_id, host_ptr, nbytes);
    return ::executorch::runtime::Error::Ok;
  }

  std::unique_ptr<HostBuffer> hb(
      HostBuffer::alias(host_ptr, nbytes, MemoryKind::HostMirror));
  if (!hb) return ::executorch::runtime::Error::MemoryAllocationFailed;
  HostBuffer* raw = hb.get();
  value_to_buffer_[value_id] = raw;
  owned_buffers_.push_back(std::move(hb));
  central_t.unsafeGetTensorImpl()->set_data(host_ptr);
  ET_LOG(Debug,
         "[mem] host_pool: bind dst=%u host_ptr=%p bytes=%zu (Aliasing wrap)",
         value_id, host_ptr, nbytes);
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error HostPoolEngine::bind_inputs(
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<const ::executorch::runtime::EValue> caller_evs,
    ::executorch::runtime::Span<const uint32_t> value_ids) {
  if (caller_evs.size() != value_ids.size())
    return ::executorch::runtime::Error::InvalidArgument;
  for (size_t i = 0; i < value_ids.size(); ++i) {
    uint32_t vid = value_ids[i];
    if (vid >= values.size())
      return ::executorch::runtime::Error::InvalidArgument;
    if (auto e = bind_one_(values[vid], vid, caller_evs[i]);
        e != ::executorch::runtime::Error::Ok)
      return e;
  }
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error HostPoolEngine::bind_outputs(
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<const ::executorch::runtime::EValue> caller_evs,
    ::executorch::runtime::Span<const uint32_t> value_ids) {
  return bind_inputs(values, caller_evs, value_ids);
}

::executorch::runtime::Error HostPoolEngine::upload_from_host(
    ::executorch::runtime::EValue& host_src_ev,
    ::executorch::runtime::EValue& dev_dst_ev,
    uint32_t dev_dst_value_id,
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  SignalGuard guard(signal);
  if (auto e = check_async_dependencies(wait_for, signal);
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }
  if (!host_src_ev.isTensor() || !dev_dst_ev.isTensor()) {
    if (signal) {
      signal->signal_failed(::executorch::runtime::Error::InvalidArgument);
    }
    return ::executorch::runtime::Error::InvalidArgument;
  }

  auto& src_t = host_src_ev.toTensor();
  auto& dst_t = dev_dst_ev.toTensor();
  size_t nbytes = src_t.nbytes();
  void* host_src_ptr = src_t.mutable_data_ptr();

  if (auto e = ::executorch::runtime::resize_tensor(dst_t, src_t.sizes());
      e != ::executorch::runtime::Error::Ok) {
    if (signal) signal->signal_failed(e);
    return e;
  }

  auto it = value_to_buffer_.find(dev_dst_value_id);
  if (it == value_to_buffer_.end()) {
    ET_LOG(Error,
           "host_pool: upload_from_host: no internal Buffer for value_id=%u",
           dev_dst_value_id);
    if (signal) signal->signal_failed(::executorch::runtime::Error::InvalidState);
    return ::executorch::runtime::Error::InvalidState;
  }
  HostBuffer* hb = it->second;
  void* dst_ptr = hb->host_ptr();
  bool was_already = (dst_ptr == host_src_ptr);
  if (!was_already && nbytes > 0) {
    std::memcpy(dst_ptr, host_src_ptr, nbytes);
  }

  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  ET_LOG(Debug,
         "[mem] host_pool: upload_from_host caller_ptr=%p bytes=%zu (%s)",
         host_src_ptr, nbytes,
         was_already ? "ptr==dst, no copy" : "memcpy");
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error HostPoolEngine::download_to_host(
    ::executorch::runtime::EValue& dev_src_ev,
    uint32_t dev_src_value_id,
    ::executorch::runtime::EValue& host_dst_ev,
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  SignalGuard guard(signal);
  if (auto e = check_async_dependencies(wait_for, signal);
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }
  if (!dev_src_ev.isTensor() || !host_dst_ev.isTensor()) {
    if (signal) {
      signal->signal_failed(::executorch::runtime::Error::InvalidArgument);
    }
    return ::executorch::runtime::Error::InvalidArgument;
  }

  auto& src_t = dev_src_ev.toTensor();
  auto& dst_t = host_dst_ev.toTensor();
  size_t nbytes = src_t.nbytes();
  void* host_dst_ptr = dst_t.mutable_data_ptr();

  if (auto e = ::executorch::runtime::resize_tensor(dst_t, src_t.sizes());
      e != ::executorch::runtime::Error::Ok) {
    if (signal) signal->signal_failed(e);
    return e;
  }

  auto it = value_to_buffer_.find(dev_src_value_id);
  if (it == value_to_buffer_.end()) {
    ET_LOG(Error,
           "host_pool: download_to_host: no internal Buffer for value_id=%u",
           dev_src_value_id);
    if (signal) signal->signal_failed(::executorch::runtime::Error::InvalidState);
    return ::executorch::runtime::Error::InvalidState;
  }
  HostBuffer* hb = it->second;
  void* src_ptr = hb->host_ptr();
  bool was_already = (src_ptr == host_dst_ptr);
  if (!was_already && nbytes > 0) {
    std::memcpy(host_dst_ptr, src_ptr, nbytes);
  }

  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  ET_LOG(Debug,
         "[mem] host_pool: download_to_host caller_ptr=%p bytes=%zu (%s)",
         host_dst_ptr, nbytes,
         was_already ? "ptr==src, no copy" : "memcpy");
  return ::executorch::runtime::Error::Ok;
}

}  // namespace native
}  // namespace backends
}  // namespace executorch
