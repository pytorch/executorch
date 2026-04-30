/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/runtimes/cpu/CpuEngine.h>

#include <unordered_map>

#include <executorch/backends/native/core/EngineUtils.h>
#include <executorch/backends/native/runtimes/cpu/CpuOpRegistry.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <string>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

namespace {

// Casts the EventStatus path uniformly. Returns Ok if event is null or
// already Complete; otherwise returns the carried error.
::executorch::runtime::Error wait_status(EventStatus s, Event* e) {
  if (s == EventStatus::Complete)
    return ::executorch::runtime::Error::Ok;
  if (e == nullptr)
    return ::executorch::runtime::Error::Ok;
  return e->error();
}

} // namespace

CpuEngine::~CpuEngine() = default;

::executorch::runtime::Result<CompiledSegment*> CpuEngine::compile_segment(
    const ::executorch::backends::portable::Graph& graph,
    ::executorch::runtime::Span<const uint32_t> instruction_indices,
    ::executorch::runtime::Span<const uint32_t> /*input_value_ids*/,
    ::executorch::runtime::Span<const uint32_t> /*output_value_ids*/,
    ::executorch::runtime::Span<const std::pair<uint32_t, uint32_t>>
        value_remap) {
  std::vector<uint32_t> idxs(
      instruction_indices.begin(), instruction_indices.end());
  std::unordered_map<uint32_t, uint32_t> remap;
  remap.reserve(value_remap.size());
  for (const auto& kv : value_remap) {
    remap.emplace(kv.first, kv.second);
  }
  auto seg = std::make_unique<CpuCompiledSegment>(
      &graph, std::move(idxs), std::move(remap));
  CompiledSegment* raw = seg.get();
  compiled_segments_.push_back(std::move(seg));
  return raw;
}

::executorch::runtime::Error CpuEngine::allocate_buffers(
    ::executorch::runtime::Span<const AllocRequest> requests,
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<AllocClaim> out_claims) {
  if (requests.size() != out_claims.size()) {
    return ::executorch::runtime::Error::InvalidArgument;
  }
  // Validate inputs.
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.value_id >= values.size() || !values[req.value_id].isTensor()) {
      ET_LOG(
          Error,
          "CpuEngine::allocate_buffers: value_id=%u missing or not a tensor",
          req.value_id);
      return ::executorch::runtime::Error::InvalidArgument;
    }
  }

  // Helper: find the host partner's pointer for a DeviceMirror request
  // by reading values[host_mirror_value_id].toTensor().data_ptr.
  // Non-null iff some prior engine (HostPool) already claimed the host
  // half and wrote its data_ptr.
  auto partner_host_ptr_for = [&](uint32_t host_vid) -> void* {
    if (host_vid == kInvalidValueId) return nullptr;
    if (host_vid >= values.size() || !values[host_vid].isTensor()) {
      return nullptr;
    }
    return values[host_vid].toTensor().mutable_data_ptr();
  };

  // CPU is the "device" runtime in this design; it claims DeviceMirror
  // and DeviceOnly requests in its own list. HostMirror / HostOnly are
  // declined (they go to HostPool).
  //
  // For DeviceMirror: alias to host partner if available (UMA collapse,
  // since CPU is always host-addressable). If partner is missing
  // (shouldn't happen with host-first ordering), fall through to fresh
  // arena allocation.

  // alias_to[i] = partner Buffer's host_ptr, or nullptr if not aliased.
  std::vector<void*> alias_ptr(requests.size(), nullptr);
  std::vector<size_t> alias_bytes(requests.size(), 0);
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.kind == MemoryKind::HostOnly ||
        req.kind == MemoryKind::HostMirror) {
      out_claims[i] = AllocClaim::Declined;
      continue;
    }
    out_claims[i] = AllocClaim::Claimed;
    if (req.kind != MemoryKind::DeviceMirror) continue;
    void* p = partner_host_ptr_for(req.host_mirror_value_id);
    if (p) {
      alias_ptr[i] = p;
      // Compute alias byte count from the host partner's tensor.
      if (req.host_mirror_value_id < values.size() &&
          values[req.host_mirror_value_id].isTensor()) {
        alias_bytes[i] =
            values[req.host_mirror_value_id].toTensor().nbytes();
      }
    }
  }

  // Per-group alias representative = biggest aliased source seen.
  std::unordered_map<int32_t, size_t> group_alias_max;
  std::unordered_map<int32_t, void*> group_alias_src;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (out_claims[i] != AllocClaim::Claimed) continue;
    if (alias_ptr[i] == nullptr) continue;
    auto it = group_alias_max.find(req.mem_obj_id);
    if (it == group_alias_max.end() || alias_bytes[i] > it->second) {
      group_alias_max[req.mem_obj_id] = alias_bytes[i];
      group_alias_src[req.mem_obj_id] = alias_ptr[i];
    }
  }
  // Extend alias to fitting non-aliased group members.
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (out_claims[i] != AllocClaim::Claimed) continue;
    if (alias_ptr[i] != nullptr) continue;
    auto it = group_alias_src.find(req.mem_obj_id);
    if (it == group_alias_src.end()) continue;
    if (values[req.value_id].toTensor().nbytes() <=
        group_alias_max[req.mem_obj_id]) {
      alias_ptr[i] = it->second;
      alias_bytes[i] = group_alias_max[req.mem_obj_id];
    }
  }

  // Hand out aliased Buffers, dedup by source pointer.
  std::unordered_map<void*, HostBuffer*> alias_dedup;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (out_claims[i] != AllocClaim::Claimed) continue;
    if (alias_ptr[i] == nullptr) continue;
    HostBuffer* hb_raw = nullptr;
    auto it = alias_dedup.find(alias_ptr[i]);
    if (it != alias_dedup.end()) {
      hb_raw = it->second;
    } else {
      std::unique_ptr<HostBuffer> hb(HostBuffer::alias(
          alias_ptr[i], alias_bytes[i], MemoryKind::DeviceMirror));
      if (!hb)
        return ::executorch::runtime::Error::MemoryAllocationFailed;
      hb_raw = hb.get();
      alias_dedup[alias_ptr[i]] = hb_raw;
      owned_buffers_.push_back(std::move(hb));
    }
    value_to_buffer_[req.value_id] = hb_raw;
    // Host-addressable: write data_ptr into central EValue.
    values[req.value_id]
        .toTensor()
        .unsafeGetTensorImpl()
        ->set_data(alias_ptr[i]);
    ET_LOG(
        Debug,
        "[mem] cpu: value_id=%u kind=%s bytes=%zu alias=%p",
        req.value_id,
        to_string(req.kind),
        alias_bytes[i],
        alias_ptr[i]);
  }

  // Arena: per-group MAX over claimed, non-aliased members; packed
  // sequentially.
  std::unordered_map<int32_t, size_t> group_caps;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (out_claims[i] != AllocClaim::Claimed) continue;
    if (alias_ptr[i] != nullptr) continue;
    size_t nb = values[req.value_id].toTensor().nbytes();
    auto& cur = group_caps[req.mem_obj_id];
    if (nb > cur) cur = nb;
  }
  constexpr size_t kAlignment = 16;
  auto align_up = [](size_t v) {
    return (v + kAlignment - 1) & ~(kAlignment - 1);
  };
  size_t arena_total = 0;
  std::unordered_map<int32_t, size_t> group_offsets;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (out_claims[i] != AllocClaim::Claimed) continue;
    if (alias_ptr[i] != nullptr) continue;
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
    ET_LOG(Debug, "[mem] cpu: arena %zu bytes at %p", arena_size_, arena_.get());
  }

  // Hand out arena Buffers. Dedup by mem_obj_id.
  std::unordered_map<int32_t, HostBuffer*> mem_obj_buffers;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (out_claims[i] != AllocClaim::Claimed) continue;
    if (alias_ptr[i] != nullptr) continue;

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
    values[req.value_id]
        .toTensor()
        .unsafeGetTensorImpl()
        ->set_data(hb_raw->host_ptr());
    ET_LOG(
        Debug,
        "[mem] cpu: value_id=%u kind=%s mem_obj_id=%d host_ptr=%p",
        req.value_id, to_string(req.kind), req.mem_obj_id, hb_raw->host_ptr());
  }
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error CpuEngine::upload_constants(
    const ::executorch::runtime::NamedDataMap& ndm,
    ::executorch::runtime::Span<const ConstRequest> requests) {
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    auto data_result = ndm.get_data(req.ndm_key);
    if (!data_result.ok()) {
      ET_LOG(
          Error,
          "CpuEngine: upload_constants NDM key '%.*s' not found (value_id=%u)",
          static_cast<int>(req.ndm_key.size()),
          req.ndm_key.data(),
          req.value_id);
      return data_result.error();
    }
    size_t bytes = data_result.get().size();
    void* ptr = const_cast<void*>(data_result.get().data());
    std::unique_ptr<HostBuffer> hb(HostBuffer::alias_ndm(
        std::move(data_result.get()), MemoryKind::DeviceOnly));
    HostBuffer* raw = hb.get();
    value_to_buffer_[req.value_id] = raw;
    ET_LOG(
        Debug,
        "[mem] cpu: upload_constants[%zu] value_id=%u key='%.*s' bytes=%zu host_ptr=%p (zero-copy NDM alias)",
        i,
        req.value_id,
        static_cast<int>(req.ndm_key.size()),
        req.ndm_key.data(),
        bytes,
        ptr);
    owned_buffers_.push_back(std::move(hb));
  }
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error CpuEngine::bind_one_(
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
    ET_LOG(
        Debug,
        "[mem] cpu: bind dst=%u host_ptr=%p bytes=%zu (re-aliased)",
        value_id,
        host_ptr,
        nbytes);
    return ::executorch::runtime::Error::Ok;
  }

  std::unique_ptr<HostBuffer> hb(
      HostBuffer::alias(host_ptr, nbytes, MemoryKind::DeviceMirror));
  if (!hb)
    return ::executorch::runtime::Error::MemoryAllocationFailed;
  HostBuffer* raw = hb.get();
  value_to_buffer_[value_id] = raw;
  owned_buffers_.push_back(std::move(hb));
  central_t.unsafeGetTensorImpl()->set_data(host_ptr);
  ET_LOG(
      Debug,
      "[mem] cpu: bind dst=%u host_ptr=%p bytes=%zu (Aliasing wrap, DeviceMirror)",
      value_id,
      host_ptr,
      nbytes);
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error CpuEngine::bind_inputs(
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

::executorch::runtime::Error CpuEngine::bind_outputs(
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<const ::executorch::runtime::EValue> caller_evs,
    ::executorch::runtime::Span<const uint32_t> value_ids) {
  return bind_inputs(values, caller_evs, value_ids);
}

std::unique_ptr<Event> CpuEngine::make_event() {
  return std::make_unique<CpuEvent>();
}

::executorch::runtime::Error CpuEngine::upload_from_host(
    ::executorch::runtime::EValue& host_src_ev,
    ::executorch::runtime::EValue& dev_dst_ev,
    uint32_t dev_dst_value_id,
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  SignalGuard guard(signal);
  if (auto e = check_dependencies_(wait_for, signal);
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }
  if (!host_src_ev.isTensor() || !dev_dst_ev.isTensor()) {
    if (signal)
      signal->signal_failed(::executorch::runtime::Error::InvalidArgument);
    return ::executorch::runtime::Error::InvalidArgument;
  }

  auto& src_t = host_src_ev.toTensor();
  auto& dst_t = dev_dst_ev.toTensor();
  void* host_src_ptr = src_t.mutable_data_ptr();
  size_t nbytes = src_t.nbytes();

  if (auto e = ::executorch::runtime::resize_tensor(dst_t, src_t.sizes());
      e != ::executorch::runtime::Error::Ok) {
    if (signal)
      signal->signal_failed(e);
    return e;
  }

  auto it = value_to_buffer_.find(dev_dst_value_id);
  if (it == value_to_buffer_.end()) {
    ET_LOG(
        Error,
        "cpu: upload_from_host: no internal Buffer for value_id=%u",
        dev_dst_value_id);
    if (signal)
      signal->signal_failed(::executorch::runtime::Error::InvalidState);
    return ::executorch::runtime::Error::InvalidState;
  }
  HostBuffer* hb = it->second;
  void* dst_ptr = hb->host_ptr();

  // Stable-alias model: Aliasing buffers' pointer was set once at
  // allocate_buffers (or bind_inputs) and equals host_src_ptr by
  // invariant.
  if (hb->mode() == HostBuffer::Mode::Aliasing) {
    if (dst_ptr != host_src_ptr) {
      ET_LOG(
          Error,
          "cpu: upload_from_host: Aliasing buffer pointer %p != "
          "host_src_ptr %p (stable-alias invariant violated)",
          dst_ptr,
          host_src_ptr);
      if (signal)
        signal->signal_failed(::executorch::runtime::Error::Internal);
      return ::executorch::runtime::Error::Internal;
    }
    if (signal) {
      signal->prepare_signal();
      signal->signal_complete();
    }
    ET_LOG(
        Debug,
        "[mem] cpu: upload_from_host caller_ptr=%p bytes=%zu (alias unchanged, metadata-only)",
        host_src_ptr,
        nbytes);
    return ::executorch::runtime::Error::Ok;
  }
  // Owned mode: memcpy bytes into the buffer's storage.
  if (nbytes > 0) {
    std::memcpy(dst_ptr, host_src_ptr, nbytes);
  }
  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  ET_LOG(
      Debug,
      "[mem] cpu: upload_from_host caller_ptr=%p bytes=%zu (memcpy into pool storage)",
      host_src_ptr,
      nbytes);
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error CpuEngine::download_to_host(
    ::executorch::runtime::EValue& dev_src_ev,
    uint32_t dev_src_value_id,
    ::executorch::runtime::EValue& host_dst_ev,
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  SignalGuard guard(signal);
  if (auto e = check_dependencies_(wait_for, signal);
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }
  if (!dev_src_ev.isTensor() || !host_dst_ev.isTensor()) {
    if (signal)
      signal->signal_failed(::executorch::runtime::Error::InvalidArgument);
    return ::executorch::runtime::Error::InvalidArgument;
  }

  auto& src_t = dev_src_ev.toTensor();
  auto& dst_t = host_dst_ev.toTensor();
  void* host_dst_ptr = dst_t.mutable_data_ptr();
  size_t nbytes = src_t.nbytes();

  if (auto e = ::executorch::runtime::resize_tensor(dst_t, src_t.sizes());
      e != ::executorch::runtime::Error::Ok) {
    if (signal)
      signal->signal_failed(e);
    return e;
  }

  auto it = value_to_buffer_.find(dev_src_value_id);
  if (it == value_to_buffer_.end()) {
    ET_LOG(
        Error,
        "cpu: download_to_host: no internal Buffer for value_id=%u",
        dev_src_value_id);
    if (signal)
      signal->signal_failed(::executorch::runtime::Error::InvalidState);
    return ::executorch::runtime::Error::InvalidState;
  }
  HostBuffer* hb = it->second;
  void* src_ptr = hb->host_ptr();

  if (hb->mode() == HostBuffer::Mode::Aliasing) {
    if (src_ptr != host_dst_ptr) {
      ET_LOG(
          Error,
          "cpu: download_to_host: Aliasing buffer pointer %p != "
          "host_dst_ptr %p (stable-alias invariant violated)",
          src_ptr,
          host_dst_ptr);
      if (signal)
        signal->signal_failed(::executorch::runtime::Error::Internal);
      return ::executorch::runtime::Error::Internal;
    }
    if (signal) {
      signal->prepare_signal();
      signal->signal_complete();
    }
    ET_LOG(
        Debug,
        "[mem] cpu: download_to_host caller_ptr=%p bytes=%zu (alias unchanged, metadata-only)",
        host_dst_ptr,
        nbytes);
    return ::executorch::runtime::Error::Ok;
  }
  if (nbytes > 0) {
    std::memcpy(host_dst_ptr, src_ptr, nbytes);
  }
  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  ET_LOG(
      Debug,
      "[mem] cpu: download_to_host caller_ptr=%p bytes=%zu (memcpy from pool storage)",
      host_dst_ptr,
      nbytes);
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error CpuEngine::check_dependencies_(
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  return check_async_dependencies(wait_for, signal);
}

::executorch::runtime::Error CpuEngine::execute(
    CompiledSegment* segment,
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  SignalGuard guard(signal);
  if (auto e = check_dependencies_(wait_for, signal);
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }

  auto* seg = static_cast<CpuCompiledSegment*>(segment);
  if (!seg) {
    return ::executorch::runtime::Error::InvalidArgument;
  }
  const auto* graph = seg->graph();
  if (!graph)
    return ::executorch::runtime::Error::InvalidState;

  // Refresh data_ptr from internal Buffer table for every tensor op-arg
  // (single source of truth: the engine's value_to_buffer_). This
  // catches any drift between TensorImpl::data_ and the bound Buffer
  // automatically — no separate verify_segment_bindings pass needed.
  auto refresh = [&](uint32_t vid_orig) {
    uint32_t vid = seg->remap(vid_orig);
    if (vid >= values.size() || !values[vid].isTensor()) return;
    auto bit = value_to_buffer_.find(vid);
    if (bit == value_to_buffer_.end()) return;
    void* p = bit->second ? bit->second->host_ptr() : nullptr;
    auto* impl = values[vid].toTensor().unsafeGetTensorImpl();
    if (impl->data() != p) impl->set_data(p);
  };

  // Drive ops via the existing portable kernel registry.
  ::executorch::runtime::KernelRuntimeContext kctx{};
  ::executorch::backends::portable::CpuGraph cpu_graph(kctx, values);

  for (uint32_t instr_idx : seg->instruction_indices()) {
    auto op = graph->get_instruction(instr_idx);
    const char* op_name = op.name();
    if (!op_name) {
      if (signal)
        signal->signal_failed(::executorch::runtime::Error::InvalidProgram);
      return ::executorch::runtime::Error::InvalidProgram;
    }
    std::string full_name = op.full_name();
    ET_LOG(
        Debug,
        "CpuEngine: instr %u op='%s' (in=%zu, out=%zu)",
        instr_idx,
        full_name.c_str(),
        op.num_inputs(),
        op.num_outputs());

    // Refresh data_ptrs for this op's args.
    for (size_t i = 0; i < op.num_inputs(); ++i) refresh(op.input(i));
    for (size_t i = 0; i < op.num_outputs(); ++i) refresh(op.output(i));

    auto handler =
        ::executorch::backends::portable::cpu_op_registry().try_get_op_fn(
            full_name);
    if (!handler) {
      ET_LOG(Error, "CpuEngine: no handler for %s", full_name.c_str());
      if (signal)
        signal->signal_failed(::executorch::runtime::Error::NotSupported);
      return ::executorch::runtime::Error::NotSupported;
    }

    std::vector<::executorch::backends::portable::ValueRef> args;
    args.reserve(op.num_inputs() + op.num_outputs());
    for (size_t i = 0; i < op.num_inputs(); ++i) {
      args.push_back(static_cast<::executorch::backends::portable::ValueRef>(
          seg->remap(op.input(i))));
    }
    for (size_t i = 0; i < op.num_outputs(); ++i) {
      args.push_back(static_cast<::executorch::backends::portable::ValueRef>(
          seg->remap(op.output(i))));
    }
    handler(cpu_graph, args);

    if (kctx.failure_state() != ::executorch::runtime::Error::Ok) {
      auto err = kctx.failure_state();
      if (signal)
        signal->signal_failed(err);
      return err;
    }
  }

  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error CpuEngine::wait(Event* event) {
  if (!event)
    return ::executorch::runtime::Error::Ok;
  event->wait_until_settled();
  return wait_status(event->status(), event);
}

} // namespace native
} // namespace backends
} // namespace executorch
