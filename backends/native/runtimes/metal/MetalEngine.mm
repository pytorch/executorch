/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/runtimes/metal/MetalEngine.h>

#include <executorch/backends/native/core/EngineUtils.h>
#include <executorch/backends/native/runtimes/metal/MetalBuffer.h>
#include <executorch/backends/native/runtimes/metal/MetalEvent.h>
#include <executorch/backends/native/runtimes/metal/MetalRuntime.h>

#include <executorch/backends/portable/runtime/metal_v2/MetalOp.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalOpRegistry.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

namespace metal_v2_ns = ::executorch::backends::metal_v2;
namespace runtime = ::executorch::runtime;

MetalEngine::MetalEngine(MetalRuntime* provider, InstanceId id)
    : provider_(provider),
      stream_(provider ? provider->stream() : nullptr),
      id_(id) {}

MetalEngine::~MetalEngine() {
  // Drain any in-flight work before tearing down owned buffers.
  drain();
  owned_buffers_.clear();
}

runtime::Error MetalEngine::check_dependencies_(
    runtime::Span<Event* const> wait_for, Event* signal) {
  for (Event* e : wait_for) {
    if (!e) continue;
    if (auto err = wait(e); err != runtime::Error::Ok) {
      if (signal) signal->signal_poisoned(err);
      return runtime::Error::Internal;
    }
  }
  return runtime::Error::Ok;
}

runtime::Result<CompiledSegment*> MetalEngine::compile_segment(
    const ::executorch::backends::portable::Graph& graph,
    runtime::Span<const uint32_t> instruction_indices,
    runtime::Span<const uint32_t> /*input_value_ids*/,
    runtime::Span<const uint32_t> /*output_value_ids*/,
    runtime::Span<const std::pair<uint32_t, uint32_t>> value_remap) {
  std::vector<uint32_t> idxs(
      instruction_indices.begin(), instruction_indices.end());
  std::unordered_map<uint32_t, uint32_t> remap;
  remap.reserve(value_remap.size());
  for (const auto& kv : value_remap) {
    remap.emplace(kv.first, kv.second);
  }
  auto seg = std::make_unique<MetalCompiledSegment>(
      &graph, std::move(idxs), std::move(remap));
  CompiledSegment* raw = seg.get();
  compiled_segments_.push_back(std::move(seg));
  return raw;
}

runtime::Error MetalEngine::allocate_buffers(
    runtime::Span<const AllocRequest> requests,
    runtime::Span<runtime::EValue> values,
    runtime::Span<AllocClaim> out_claims) {
  if (!stream_) return runtime::Error::InvalidState;
  if (requests.size() != out_claims.size()) {
    return runtime::Error::InvalidArgument;
  }
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.value_id >= values.size() || !values[req.value_id].isTensor()) {
      ET_LOG(Error,
             "MetalEngine::allocate_buffers: value_id=%u missing or not a tensor",
             req.value_id);
      return runtime::Error::InvalidArgument;
    }
  }

  // Helper: read host partner's data_ptr from the central EValue (set
  // by HostPool's prior allocate_buffers pass).
  auto partner_host_ptr_for = [&](uint32_t host_vid) -> void* {
    if (host_vid == kInvalidValueId) return nullptr;
    if (host_vid >= values.size() || !values[host_vid].isTensor()) {
      return nullptr;
    }
    return values[host_vid].toTensor().mutable_data_ptr();
  };

  // Metal claims DeviceMirror and DeviceOnly. Declines HostMirror /
  // HostOnly (HostPool handles those).
  //
  // For DeviceMirror: try zero-copy alias collapse with the host
  // partner's pointer (UMA fast path); fall back to fresh pool
  // allocation.

  // For each mem_obj_id group, allocate ONE Buffer at MAX nbytes
  // across all sharers.
  auto mem_obj_caps = compute_mem_obj_capacity(
      requests,
      ::executorch::runtime::Span<const ::executorch::runtime::EValue>(
          values.data(), values.size()));
  std::unordered_map<int32_t, MetalBuffer*> mem_obj_buffers;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];

    if (req.kind == MemoryKind::HostOnly ||
        req.kind == MemoryKind::HostMirror) {
      out_claims[i] = AllocClaim::Declined;
      continue;
    }
    out_claims[i] = AllocClaim::Claimed;

    // mem_obj_id cache hit.
    auto it = mem_obj_buffers.find(req.mem_obj_id);
    if (it != mem_obj_buffers.end()) {
      MetalBuffer* shared = it->second;
      value_to_buffer_[req.value_id] = shared;
      void* hp = shared->host_ptr();
      if (hp) {
        values[req.value_id]
            .toTensor()
            .unsafeGetTensorImpl()
            ->set_data(hp);
      }
      ET_LOG(Debug,
             "[mem] metal: value_id=%u shares mem_obj_id=%d -> Buffer host_ptr=%p",
             req.value_id, req.mem_obj_id, hp);
      continue;
    }

    MetalBuffer* buf = nullptr;

    // DeviceMirror with a host-addressable partner: try zero-copy alias.
    void* partner_p = nullptr;
    size_t partner_bytes = 0;
    if (req.kind == MemoryKind::DeviceMirror) {
      partner_p = partner_host_ptr_for(req.host_mirror_value_id);
      if (partner_p && req.host_mirror_value_id < values.size() &&
          values[req.host_mirror_value_id].isTensor()) {
        partner_bytes =
            values[req.host_mirror_value_id].toTensor().nbytes();
      }
    }
    if (partner_p &&
        stream_->registerExternalBuffer(
            partner_p, partner_bytes, /*strict_zero_copy=*/true)) {
      buf = MetalBuffer::alias(
          stream_, partner_p, partner_bytes, MemoryKind::DeviceMirror);
      ET_LOG(Debug,
             "[mem] metal: value_id=%u kind=DeviceMirror bytes=%zu host_partner_vid=%u (zero-copy alias / UMA collapse)",
             req.value_id, partner_bytes, req.host_mirror_value_id);
      owned_buffers_.emplace_back(buf);
    }

    if (!buf) {
      // Fresh pool allocation.
      size_t nbytes = mem_obj_caps[req.mem_obj_id];
      if (nbytes == 0) nbytes = values[req.value_id].toTensor().nbytes();
      void* ptr = stream_->alloc(nbytes);
      if (!ptr) return runtime::Error::MemoryAllocationFailed;
      (void)stream_->bufferForPtr(ptr, nbytes);
      buf = MetalBuffer::allocate(stream_, ptr, nbytes, req.kind);
      ET_LOG(Debug,
             "[mem] metal: value_id=%u kind=%s bytes=%zu host_ptr=%p (mem_obj group capacity)",
             req.value_id, to_string(req.kind), nbytes, ptr);
      owned_buffers_.emplace_back(buf);
    }

    value_to_buffer_[req.value_id] = buf;
    mem_obj_buffers[req.mem_obj_id] = buf;
    void* hp = buf->host_ptr();
    if (hp) {
      values[req.value_id]
          .toTensor()
          .unsafeGetTensorImpl()
          ->set_data(hp);
    }
  }
  return runtime::Error::Ok;
}

runtime::Error MetalEngine::upload_constants(
    const runtime::NamedDataMap& ndm,
    runtime::Span<const ConstRequest> requests) {
  if (!stream_) return runtime::Error::InvalidState;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    auto fb_result = ndm.get_data(req.ndm_key);
    if (!fb_result.ok()) {
      ET_LOG(
          Error,
          "MetalEngine: upload_constants NDM key '%.*s' not found (value_id=%u)",
          static_cast<int>(req.ndm_key.size()),
          req.ndm_key.data(),
          req.value_id);
      return fb_result.error();
    }
    runtime::FreeableBuffer fb = std::move(fb_result.get());
    void* ptr = const_cast<void*>(fb.data());
    size_t bytes = fb.size();
    stream_->registerExternalBuffer(ptr, bytes);
    (void)stream_->bufferForPtr(ptr, bytes);
    auto* buf = MetalBuffer::alias_ndm(
        stream_, std::move(fb), MemoryKind::DeviceOnly);
    owned_buffers_.emplace_back(buf);
    value_to_buffer_[req.value_id] = buf;
    ET_LOG(
        Debug,
        "[mem] metal: upload_constants[%zu] value_id=%u key='%.*s' bytes=%zu host_ptr=%p (zero-copy NDM alias)",
        i,
        req.value_id,
        static_cast<int>(req.ndm_key.size()),
        req.ndm_key.data(),
        bytes,
        ptr);
  }
  return runtime::Error::Ok;
}

runtime::Error MetalEngine::bind_one_(
    runtime::EValue& central_ev,
    uint32_t value_id,
    const runtime::EValue& caller_ev) {
  if (!stream_) return runtime::Error::InvalidState;
  if (!caller_ev.isTensor() || !central_ev.isTensor())
    return runtime::Error::InvalidArgument;
  auto& caller_t = caller_ev.toTensor();
  void* host_ptr = caller_t.mutable_data_ptr();
  size_t nbytes = caller_t.nbytes();
  if (!host_ptr) return runtime::Error::InvalidArgument;

  auto& central_t = central_ev.toTensor();
  if (auto e = runtime::resize_tensor(central_t, caller_t.sizes());
      e != runtime::Error::Ok) {
    return e;
  }

  auto it = value_to_buffer_.find(value_id);
  if (it != value_to_buffer_.end()) {
    MetalBuffer* mb = it->second;
    if (mb->mode() == MetalBuffer::Mode::Aliasing) {
      if (!stream_->registerExternalBuffer(
              host_ptr, nbytes, /*strict_zero_copy=*/true)) {
        ET_LOG(Error,
               "metal: bind: re-bind of dst=%u to caller_ptr=%p "
               "(bytes=%zu) refused zero-copy on Aliasing buffer",
               value_id, host_ptr, nbytes);
        return runtime::Error::NotSupported;
      }
      mb->re_alias(host_ptr, nbytes);
      central_t.unsafeGetTensorImpl()->set_data(host_ptr);
      ET_LOG(Debug,
             "[mem] metal: bind dst=%u host_ptr=%p bytes=%zu (re-aliased)",
             value_id, host_ptr, nbytes);
      return runtime::Error::Ok;
    }
    void* dst = mb->host_ptr();
    if (dst && nbytes > 0) std::memcpy(dst, host_ptr, nbytes);
    if (dst) {
      central_t.unsafeGetTensorImpl()->set_data(dst);
    }
    ET_LOG(Debug,
           "[mem] metal: bind dst=%u host_ptr=%p bytes=%zu (memcpy into pool storage)",
           value_id, host_ptr, nbytes);
    return runtime::Error::Ok;
  }

  if (stream_->registerExternalBuffer(
          host_ptr, nbytes, /*strict_zero_copy=*/true)) {
    auto* mb = MetalBuffer::alias(
        stream_, host_ptr, nbytes, MemoryKind::DeviceMirror);
    value_to_buffer_[value_id] = mb;
    owned_buffers_.emplace_back(mb);
    central_t.unsafeGetTensorImpl()->set_data(host_ptr);
    ET_LOG(Debug,
           "[mem] metal: bind dst=%u host_ptr=%p bytes=%zu (Aliasing wrap, DeviceMirror)",
           value_id, host_ptr, nbytes);
    return runtime::Error::Ok;
  }
  void* ptr = stream_->alloc(nbytes);
  if (!ptr) return runtime::Error::MemoryAllocationFailed;
  (void)stream_->bufferForPtr(ptr, nbytes);
  auto* mb = MetalBuffer::allocate(
      stream_, ptr, nbytes, MemoryKind::DeviceMirror);
  if (nbytes > 0) std::memcpy(ptr, host_ptr, nbytes);
  value_to_buffer_[value_id] = mb;
  owned_buffers_.emplace_back(mb);
  central_t.unsafeGetTensorImpl()->set_data(ptr);
  ET_LOG(Debug,
         "[mem] metal: bind dst=%u host_ptr=%p bytes=%zu (Owned pool, alias refused)",
         value_id, host_ptr, nbytes);
  return runtime::Error::Ok;
}

runtime::Error MetalEngine::bind_inputs(
    runtime::Span<runtime::EValue> values,
    runtime::Span<const runtime::EValue> caller_evs,
    runtime::Span<const uint32_t> value_ids) {
  if (caller_evs.size() != value_ids.size())
    return runtime::Error::InvalidArgument;
  for (size_t i = 0; i < value_ids.size(); ++i) {
    uint32_t vid = value_ids[i];
    if (vid >= values.size())
      return runtime::Error::InvalidArgument;
    if (auto e = bind_one_(values[vid], vid, caller_evs[i]);
        e != runtime::Error::Ok)
      return e;
  }
  return runtime::Error::Ok;
}

runtime::Error MetalEngine::bind_outputs(
    runtime::Span<runtime::EValue> values,
    runtime::Span<const runtime::EValue> caller_evs,
    runtime::Span<const uint32_t> value_ids) {
  return bind_inputs(values, caller_evs, value_ids);
}

std::unique_ptr<Event> MetalEngine::make_event() {
  return std::make_unique<MetalEvent>();
}

runtime::Error MetalEngine::upload_from_host(
    runtime::EValue& host_src_ev,
    runtime::EValue& dev_dst_ev,
    uint32_t dev_dst_value_id,
    runtime::Span<Event* const> wait_for,
    Event* signal) {
  SignalGuard guard(signal);
  if (auto e = check_dependencies_(wait_for, signal);
      e != runtime::Error::Ok) {
    return e;
  }

  if (!host_src_ev.isTensor() || !dev_dst_ev.isTensor()) {
    if (signal) signal->signal_failed(runtime::Error::InvalidArgument);
    return runtime::Error::InvalidArgument;
  }

  auto& src_t = host_src_ev.toTensor();
  auto& dst_t = dev_dst_ev.toTensor();
  void* host_src_ptr = src_t.mutable_data_ptr();
  size_t nbytes = src_t.nbytes();

  if (auto e = runtime::resize_tensor(dst_t, src_t.sizes());
      e != runtime::Error::Ok) {
    if (signal) signal->signal_failed(e);
    return e;
  }

  auto it = value_to_buffer_.find(dev_dst_value_id);
  if (it == value_to_buffer_.end()) {
    ET_LOG(Error,
           "metal: upload_from_host: no internal Buffer for value_id=%u",
           dev_dst_value_id);
    if (signal) signal->signal_failed(runtime::Error::InvalidState);
    return runtime::Error::InvalidState;
  }
  auto* mb = it->second;

  if (mb->mode() == MetalBuffer::Mode::Aliasing) {
    if (mb->host_ptr() != host_src_ptr) {
      ET_LOG(Error,
             "metal: upload_from_host: Aliasing buffer pointer %p != "
             "host_src_ptr %p (stable-alias invariant violated)",
             mb->host_ptr(), host_src_ptr);
      if (signal) signal->signal_failed(runtime::Error::Internal);
      return runtime::Error::Internal;
    }
    if (signal) {
      signal->prepare_signal();
      signal->signal_complete();
    }
    ET_LOG(Debug,
           "[mem] metal: upload_from_host caller_ptr=%p bytes=%zu (alias unchanged, metadata-only)",
           host_src_ptr, nbytes);
    return runtime::Error::Ok;
  }

  void* dst_ptr = mb->host_ptr();
  if (!dst_ptr) {
    ET_LOG(Error,
           "metal: upload_from_host: VRAM-only destination not yet supported");
    if (signal) signal->signal_failed(runtime::Error::NotImplemented);
    return runtime::Error::NotImplemented;
  }
  if (nbytes > 0) {
    std::memcpy(dst_ptr, host_src_ptr, nbytes);
  }
  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  ET_LOG(Debug,
         "[mem] metal: upload_from_host caller_ptr=%p bytes=%zu (memcpy into pool storage)",
         host_src_ptr, nbytes);
  return runtime::Error::Ok;
}

runtime::Error MetalEngine::download_to_host(
    runtime::EValue& dev_src_ev,
    uint32_t dev_src_value_id,
    runtime::EValue& host_dst_ev,
    runtime::Span<Event* const> wait_for,
    Event* signal) {
  SignalGuard guard(signal);
  if (auto e = check_dependencies_(wait_for, signal);
      e != runtime::Error::Ok) {
    return e;
  }

  if (!dev_src_ev.isTensor() || !host_dst_ev.isTensor()) {
    if (signal) signal->signal_failed(runtime::Error::InvalidArgument);
    return runtime::Error::InvalidArgument;
  }

  auto& src_t = dev_src_ev.toTensor();
  auto& dst_t = host_dst_ev.toTensor();
  void* host_dst_ptr = dst_t.mutable_data_ptr();
  size_t nbytes = src_t.nbytes();

  if (auto e = runtime::resize_tensor(dst_t, src_t.sizes());
      e != runtime::Error::Ok) {
    if (signal) signal->signal_failed(e);
    return e;
  }

  auto it = value_to_buffer_.find(dev_src_value_id);
  if (it == value_to_buffer_.end()) {
    ET_LOG(Error,
           "metal: download_to_host: no internal Buffer for value_id=%u",
           dev_src_value_id);
    if (signal) signal->signal_failed(runtime::Error::InvalidState);
    return runtime::Error::InvalidState;
  }
  auto* mb = it->second;

  if (mb->host_ptr() == host_dst_ptr) {
    if (signal) {
      signal->prepare_signal();
      signal->signal_complete();
    }
    ET_LOG(Debug,
           "[mem] metal: download_to_host caller_ptr=%p bytes=%zu (alias unchanged)",
           host_dst_ptr, nbytes);
    return runtime::Error::Ok;
  }

  void* src_ptr = mb->host_ptr();
  if (!src_ptr) {
    ET_LOG(Error,
           "metal: download_to_host: VRAM-only source not yet supported");
    if (signal) signal->signal_failed(runtime::Error::NotImplemented);
    return runtime::Error::NotImplemented;
  }
  std::memcpy(host_dst_ptr, src_ptr, nbytes);
  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  ET_LOG(Debug,
         "[mem] metal: download_to_host caller_ptr=%p bytes=%zu (memcpy)",
         host_dst_ptr, nbytes);
  return runtime::Error::Ok;
}

runtime::Error MetalEngine::execute(
    CompiledSegment* segment,
    runtime::Span<runtime::EValue> values,
    runtime::Span<Event* const> wait_for,
    Event* signal) {
  SignalGuard guard(signal);
  if (auto e = check_dependencies_(wait_for, signal);
      e != runtime::Error::Ok) {
    return e;
  }

  auto* seg = static_cast<MetalCompiledSegment*>(segment);
  if (!seg) return runtime::Error::InvalidArgument;
  const auto* graph = seg->graph();
  if (!graph) return runtime::Error::InvalidState;

  // Refresh data_ptr from internal Buffer table for every tensor op-arg.
  // Single source of truth: value_to_buffer_'s host_ptr(). On Apple
  // Silicon UMA, this equals the MetalBuffer's contents pointer, which
  // doubles as the lookup key for stream->bufferForPtr() inside ops.
  auto refresh = [&](uint32_t vid_orig) {
    uint32_t vid = seg->remap(vid_orig);
    if (vid >= values.size() || !values[vid].isTensor()) return;
    auto bit = value_to_buffer_.find(vid);
    if (bit == value_to_buffer_.end()) return;
    void* p = bit->second ? bit->second->host_ptr() : nullptr;
    auto* impl = values[vid].toTensor().unsafeGetTensorImpl();
    if (impl->data() != p) impl->set_data(p);
  };

  auto& registry = metal_v2_ns::MetalOpRegistry::shared();
  for (uint32_t instr_idx : seg->instruction_indices()) {
    auto op = graph->get_instruction(instr_idx);
    const char* op_name = op.name();
    if (!op_name) {
      if (signal) signal->signal_failed(runtime::Error::InvalidProgram);
      return runtime::Error::InvalidProgram;
    }
    metal_v2_ns::MetalOp* metal_op = registry.get(op_name);
    if (!metal_op) {
      ET_LOG(Error, "MetalEngine: op '%s' not in MetalOpRegistry", op_name);
      if (signal) signal->signal_failed(runtime::Error::NotSupported);
      return runtime::Error::NotSupported;
    }

    ET_LOG(Debug, "MetalEngine: instr %u op='%s' (in=%zu, out=%zu)",
           instr_idx, op_name, op.num_inputs(), op.num_outputs());

    // Refresh data_ptrs for this op's args.
    for (size_t i = 0; i < op.num_inputs(); ++i) refresh(op.input(i));
    for (size_t i = 0; i < op.num_outputs(); ++i) refresh(op.output(i));

    // Build EValue* vectors for the op (with value_remap applied).
    std::vector<runtime::EValue*> ins;
    std::vector<runtime::EValue*> outs;
    ins.reserve(op.num_inputs());
    outs.reserve(op.num_outputs());
    for (size_t i = 0; i < op.num_inputs(); ++i) {
      uint32_t vid = seg->remap(op.input(i));
      if (vid < values.size()) ins.push_back(&values[vid]);
    }
    for (size_t i = 0; i < op.num_outputs(); ++i) {
      uint32_t vid = seg->remap(op.output(i));
      if (vid < values.size()) outs.push_back(&values[vid]);
    }

    metal_op->dispatch(
        stream_,
        runtime::Span<runtime::EValue*>(ins.data(), ins.size()),
        runtime::Span<runtime::EValue*>(outs.data(), outs.size()));
  }

  stream_->sync();

  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  return runtime::Error::Ok;
}

runtime::Error MetalEngine::wait(Event* event) {
  if (!event) return runtime::Error::Ok;
  while (true) {
    auto s = event->status();
    if (s == EventStatus::Complete) return runtime::Error::Ok;
    if (s == EventStatus::Failed || s == EventStatus::Poisoned) {
      return event->error();
    }
  }
}

void MetalEngine::drain() {
  if (stream_) stream_->sync();
}

}  // namespace native
}  // namespace backends
}  // namespace executorch
