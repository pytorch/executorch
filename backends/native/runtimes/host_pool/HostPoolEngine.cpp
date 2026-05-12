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
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<const AllocRequest> requests,
    ::executorch::runtime::Span<AllocClaim> out_claims) {
  if (requests.size() != out_claims.size()) {
    return ::executorch::runtime::Error::InvalidArgument;
  }
  // Sanity-check inputs first.
  for (const auto& req : requests) {
    if (req.value_id >= values.size() || !values[req.value_id].isTensor()) {
      ET_LOG(
          Error,
          "HostPoolEngine::allocate_buffers: value_id=%u missing or not a tensor",
          req.value_id);
      return ::executorch::runtime::Error::InvalidArgument;
    }
  }

  // HostPool dispatches on MemoryKind alone:
  //   HostExtern  -> claim, defer storage to per-execute bind_inputs /
  //                  bind_outputs (Aliasing wrapper around caller ptr).
  //   HostMirror  -> claim, allocate from arena.
  //   DeviceMirror / DeviceOnly -> decline.

  // First pass: mark claim status, identify which requests need arena.
  // Note on mem_obj_id < 0: the planner did not assign a slot (the vid
  // is unplanned / dynamic-unbounded). HostPool claims ownership but
  // defers allocation to Engine::resize_tensor on first use.
  std::vector<bool> needs_arena(requests.size(), false);
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.kind == MemoryKind::DeviceMirror ||
        req.kind == MemoryKind::DeviceOnly) {
      out_claims[i] = AllocClaim::Declined;
      continue;
    }
    out_claims[i] = AllocClaim::Claimed;
    if (req.kind == MemoryKind::HostExtern) {
      // Caller-owned storage: defer to per-execute bind_inputs /
      // bind_outputs, which allocates an Aliasing HostBuffer wrapping
      // caller's pointer. No arena slot.
      ET_LOG(
          Debug,
          "[mem] host_pool: value_id=%u kind=%s (deferred to bind_io)",
          req.value_id,
          to_string(req.kind));
      continue;
    }
    if (req.mem_obj_id < 0) {
      // Unplanned / dynamic: defer to resize_tensor. Record ownership
      // by inserting a null placeholder so resize_tensor knows this
      // vid is ours.
      value_to_buffer_[req.value_id] = nullptr;
      ET_LOG(
          Debug,
          "[mem] host_pool: value_id=%u kind=%s mem_obj_id=%d (deferred to resize_tensor)",
          req.value_id,
          to_string(req.kind),
          req.mem_obj_id);
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
    if (!needs_arena[i])
      continue;
    const auto& req = requests[i];
    size_t nb = values[req.value_id].toTensor().nbytes();
    auto& cur = group_caps[req.mem_obj_id];
    if (nb > cur)
      cur = nb;
  }

  // Lay out arena slots; one per (non-deferred) group.
  std::unordered_map<int32_t, size_t> group_offsets;
  size_t arena_total = 0;
  for (size_t i = 0; i < requests.size(); ++i) {
    if (!needs_arena[i])
      continue;
    const auto& req = requests[i];
    if (req.mem_obj_id < 0)
      continue; // deferred
    if (group_offsets.count(req.mem_obj_id))
      continue;
    size_t off = align_up(arena_total);
    group_offsets[req.mem_obj_id] = off;
    arena_total = off + group_caps[req.mem_obj_id];
  }

  if (arena_total > 0) {
    size_t alloc_bytes = align_up(arena_total);
    arena_.reset(
        static_cast<uint8_t*>(std::aligned_alloc(kAlignment, alloc_bytes)));
    if (!arena_)
      return ::executorch::runtime::Error::MemoryAllocationFailed;
    arena_size_ = alloc_bytes;
    ET_LOG(
        Debug,
        "[mem] host_pool: arena allocated %zu bytes at %p",
        arena_size_,
        arena_.get());
  }

  // Hand out Aliasing HostBuffers. Dedup per mem_obj_id.
  std::unordered_map<int32_t, HostBuffer*> mem_obj_buffers;
  for (size_t i = 0; i < requests.size(); ++i) {
    if (!needs_arena[i])
      continue;
    const auto& req = requests[i];
    if (req.mem_obj_id < 0)
      continue; // deferred (no arena slot)

    HostBuffer* hb_raw = nullptr;
    auto it = mem_obj_buffers.find(req.mem_obj_id);
    if (it != mem_obj_buffers.end()) {
      hb_raw = it->second;
    } else {
      size_t cap = group_caps[req.mem_obj_id];
      size_t off = group_offsets[req.mem_obj_id];
      std::unique_ptr<HostBuffer> hb(
          HostBuffer::alias(arena_.get() + off, cap, req.kind));
      if (!hb)
        return ::executorch::runtime::Error::MemoryAllocationFailed;
      hb_raw = hb.get();
      mem_obj_buffers[req.mem_obj_id] = hb_raw;
      owned_buffers_.push_back(std::move(hb));
    }
    value_to_buffer_[req.value_id] = hb_raw;
    // Host-addressable: write data_ptr into central EValue so other
    // engines (DeviceMirror partners) and the executor see it.
    values[req.value_id].toTensor().unsafeGetTensorImpl()->set_data(
        hb_raw->host_ptr());
    ET_LOG(
        Debug,
        "[mem] host_pool: value_id=%u kind=%s mem_obj_id=%d host_ptr=%p",
        req.value_id,
        to_string(req.kind),
        req.mem_obj_id,
        hb_raw->host_ptr());
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
  if (!host_ptr)
    return ::executorch::runtime::Error::InvalidArgument;

  // Resize central TensorImpl to caller's actual shape.
  auto& central_t = central_ev.toTensor();
  if (auto e =
          ::executorch::runtime::resize_tensor(central_t, caller_t.sizes());
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }

  auto it = value_to_buffer_.find(value_id);
  if (it != value_to_buffer_.end()) {
    HostBuffer* hb = it->second;
    hb->re_alias(host_ptr, nbytes);
    central_t.unsafeGetTensorImpl()->set_data(host_ptr);
    ET_LOG(
        Debug,
        "[mem] host_pool: bind dst=%u host_ptr=%p bytes=%zu (re-aliased)",
        value_id,
        host_ptr,
        nbytes);
    return ::executorch::runtime::Error::Ok;
  }

  std::unique_ptr<HostBuffer> hb(
      HostBuffer::alias(host_ptr, nbytes, MemoryKind::HostMirror));
  if (!hb)
    return ::executorch::runtime::Error::MemoryAllocationFailed;
  HostBuffer* raw = hb.get();
  value_to_buffer_[value_id] = raw;
  owned_buffers_.push_back(std::move(hb));
  central_t.unsafeGetTensorImpl()->set_data(host_ptr);
  ET_LOG(
      Debug,
      "[mem] host_pool: bind dst=%u host_ptr=%p bytes=%zu (Aliasing wrap)",
      value_id,
      host_ptr,
      nbytes);
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error HostPoolEngine::set_io_bindings(
    ::executorch::runtime::Span<const InputBinding> graph_inputs,
    ::executorch::runtime::Span<const OutputBinding> graph_outputs) {
  io_input_bindings_.clear();
  io_input_bindings_.reserve(graph_inputs.size());
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    io_input_bindings_.push_back(
        {static_cast<uint32_t>(i), graph_inputs[i].value_id});
  }
  io_output_bindings_.clear();
  io_output_bindings_.reserve(graph_outputs.size());
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    io_output_bindings_.push_back(
        {static_cast<uint32_t>(i), graph_outputs[i].value_id});
  }
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error HostPoolEngine::resize_tensor(
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    uint32_t value_id,
    ::executorch::runtime::ArrayRef<::executorch::aten::SizesType> new_sizes) {
  if (value_id >= values.size() || !values[value_id].isTensor()) {
    return ::executorch::runtime::Error::InvalidArgument;
  }
  auto it = value_to_buffer_.find(value_id);
  if (it == value_to_buffer_.end()) {
    ET_LOG(
        Error,
        "host_pool: resize_tensor: vid=%u not owned by HostPool",
        value_id);
    return ::executorch::runtime::Error::InvalidArgument;
  }

  auto& central_t = values[value_id].toTensor();
  size_t new_nbytes = bytes_for_sizes(central_t.scalar_type(), new_sizes);

  HostBuffer* hb = it->second;
  bool need_alloc = (hb == nullptr) || (hb->size_bytes() < new_nbytes);
  if (need_alloc) {
    constexpr size_t kAlignment = 16;
    std::unique_ptr<HostBuffer> fresh(
        HostBuffer::allocate(new_nbytes, kAlignment, MemoryKind::HostMirror));
    if (!fresh) {
      return ::executorch::runtime::Error::MemoryAllocationFailed;
    }
    hb = fresh.get();
    value_to_buffer_[value_id] = hb;
    owned_buffers_.push_back(std::move(fresh));
    ET_LOG(
        Debug,
        "[mem] host_pool: resize_tensor vid=%u allocated %zu bytes at %p",
        value_id,
        new_nbytes,
        hb->host_ptr());
  }

  // Update central TensorImpl's shape and data pointer.
  if (auto e = ::executorch::runtime::resize_tensor(central_t, new_sizes);
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }
  central_t.unsafeGetTensorImpl()->set_data(hb->host_ptr());
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error HostPoolEngine::bind_inputs(
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<const ::executorch::runtime::EValue* const>
input_args) {
  for (const auto& b : io_input_bindings_) {
    if (b.graph_idx >= input_args.size())
      continue;
    if (b.internal_vid >= values.size())
      return ::executorch::runtime::Error::InvalidArgument;
    if (!input_args[b.graph_idx])
      continue;
    if (auto e = bind_one_(
            values[b.internal_vid], b.internal_vid, *input_args[b.graph_idx]);
        e != ::executorch::runtime::Error::Ok)
      return e;
  }
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error HostPoolEngine::bind_outputs(
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<::executorch::runtime::EValue* const>
        output_args) {
  for (const auto& b : io_output_bindings_) {
    if (b.graph_idx >= output_args.size())
      continue;
    if (b.internal_vid >= values.size())
      return ::executorch::runtime::Error::InvalidArgument;
    if (!output_args[b.graph_idx])
      continue;
    if (auto e = bind_one_(
            values[b.internal_vid], b.internal_vid, *output_args[b.graph_idx]);
        e != ::executorch::runtime::Error::Ok)
      return e;
  }
  return ::executorch::runtime::Error::Ok;
}

} // namespace native
} // namespace backends
} // namespace executorch
