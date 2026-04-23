/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/portable/runtime_v2/cpu/CpuInstance.h>

#include <unordered_map>

#include <executorch/backends/portable/runtime_v2/cpu/CpuOpRegistry.h>
#include <executorch/backends/portable/runtime_v2/api/InstanceUtils.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <string>
#include <vector>

namespace executorch {
namespace backends {
namespace portable_v2 {

namespace {

// Casts the EventStatus path uniformly. Returns Ok if event is null or
// already Complete; otherwise returns the carried error.
::executorch::runtime::Error wait_status(EventStatus s, Event* e) {
  if (s == EventStatus::Complete) return ::executorch::runtime::Error::Ok;
  if (e == nullptr) return ::executorch::runtime::Error::Ok;
  return e->error();
}

}  // namespace

CpuInstance::~CpuInstance() = default;

::executorch::runtime::Result<CompiledSegment*> CpuInstance::compile_segment(
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
  auto seg = std::make_unique<CpuCompiledSegment>(&graph, std::move(idxs),
                                                  std::move(remap));
  CompiledSegment* raw = seg.get();
  compiled_segments_.push_back(std::move(seg));
  return raw;
}

::executorch::runtime::Error CpuInstance::allocate_all(
    ::executorch::runtime::Span<const AllocRequest> requests,
    ::executorch::runtime::Span<const ::executorch::runtime::EValue> values,
    ::executorch::runtime::Span<Buffer*> out_buffers) {
  if (requests.size() != out_buffers.size()) {
    return ::executorch::runtime::Error::InvalidArgument;
  }
  // CPU has unified memory: simple per-request allocation.
  // Default 16-byte alignment for SIMD.
  // host_alias would only be set for cross-runtime synthetic values whose
  // CONSUMER is on CPU; in v1 (host=CPU), all synthetic mirrors live on
  // device, not on host, so host_alias is always null here. We still
  // honor it defensively so future "device produces, host consumes"
  // cases can re-use the same plumbing.
  // Dedup by mem_obj_id (set by the AOT memory planner): multiple
  // AllocRequests sharing the same mem_obj_id (>=0) refer to the SAME
  // physical slot — e.g. a buffer placeholder and its mutation result
  // (in-place buffer mutation), or two values with non-overlapping
  // lifetimes that the planner reused. Allocate one Buffer per group;
  // every sharer points at the same data_ptr.
  std::unordered_map<int32_t, Buffer*> mem_obj_buffers;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.value_id >= values.size() ||
        !values[req.value_id].isTensor()) {
      ET_LOG(Error,
             "CpuInstance::allocate_all: value_id=%u missing or not a tensor",
             req.value_id);
      return ::executorch::runtime::Error::InvalidArgument;
    }

    if (req.mem_obj_id >= 0) {
      auto it = mem_obj_buffers.find(req.mem_obj_id);
      if (it != mem_obj_buffers.end()) {
        out_buffers[i] = it->second;
        ET_LOG(Debug,
               "[mem] cpu: allocate_all value_id=%u shares mem_obj_id=%d -> Buffer host_ptr=%p",
               req.value_id, req.mem_obj_id, it->second->host_ptr());
        continue;
      }
    }

    const auto& tensor = values[req.value_id].toTensor();
    size_t nbytes = tensor.nbytes();
    Buffer* buf = nullptr;

    // Zero-copy alias path: source already host-addressable, just wrap.
    if (req.host_alias && req.host_alias->host_ptr()) {
      void* p = req.host_alias->host_ptr();
      std::unique_ptr<HostBuffer> hb(HostBuffer::alias(p, nbytes));
      buf = hb.get();
      ET_LOG(Debug,
             "[mem] cpu: allocate_all value_id=%u bytes=%zu host_alias=%p (zero-copy alias)",
             req.value_id, nbytes, p);
      owned_buffers_.push_back(std::move(hb));
    } else {
      // Fresh allocation.
      std::unique_ptr<HostBuffer> hb(HostBuffer::allocate(nbytes, /*alignment=*/16));
      if (!hb) return ::executorch::runtime::Error::MemoryAllocationFailed;
      buf = hb.get();
      ET_LOG(Debug,
             "[mem] cpu: allocate_all value_id=%u bytes=%zu host_ptr=%p",
             req.value_id, nbytes, buf->host_ptr());
      owned_buffers_.push_back(std::move(hb));
    }
    out_buffers[i] = buf;
    if (req.mem_obj_id >= 0) {
      mem_obj_buffers[req.mem_obj_id] = buf;
    }
  }
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Result<Buffer*> CpuInstance::upload_constant(
    const ::executorch::runtime::NamedDataMap& ndm,
    std::string_view key) {
  // CPU: zero-copy alias the FreeableBuffer's data. The HostBuffer
  // takes ownership of the FreeableBuffer (frees it in destructor).
  auto data_result = ndm.get_data(key);
  if (!data_result.ok()) {
    ET_LOG(Error,
           "CpuInstance: NDM key not found for constant upload");
    return data_result.error();
  }
  size_t bytes = data_result.get().size();
  void* ptr = const_cast<void*>(data_result.get().data());
  std::unique_ptr<HostBuffer> hb(
      HostBuffer::alias_ndm(std::move(data_result.get())));
  Buffer* raw = hb.get();
  ET_LOG(Debug,
         "[mem] cpu: upload_constant key='%.*s' bytes=%zu host_ptr=%p (zero-copy NDM alias)",
         static_cast<int>(key.size()), key.data(), bytes, ptr);
  owned_buffers_.push_back(std::move(hb));
  return raw;
}

std::unique_ptr<Event> CpuInstance::make_event() {
  return std::make_unique<CpuEvent>();
}

::executorch::runtime::Error CpuInstance::upload_from_host(
    ::executorch::runtime::EValue& host_src_ev,
    void* host_src_ptr,
    ::executorch::runtime::EValue& dev_dst_ev,
    Buffer* dev_dst_buf,
    QueueKind /*queue*/,
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  if (auto e = check_dependencies_(wait_for, signal);
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }
  if (!host_src_ptr || !dev_dst_buf || !host_src_ev.isTensor() ||
      !dev_dst_ev.isTensor()) {
    if (signal) signal->signal_failed(::executorch::runtime::Error::InvalidArgument);
    return ::executorch::runtime::Error::InvalidArgument;
  }

  auto& src_t = host_src_ev.toTensor();
  auto& dst_t = dev_dst_ev.toTensor();
  size_t nbytes = src_t.nbytes();

  // Propagate shape (per the shape-on-event contract).
  if (auto e = ::executorch::runtime::resize_tensor(dst_t, src_t.sizes());
      e != ::executorch::runtime::Error::Ok) {
    if (signal) signal->signal_failed(e);
    return e;
  }

  // Re-alias the destination HostBuffer to point at host_src_ptr.
  // Idempotent if already pointing there. Frees any prior Owned storage.
  auto* hb = static_cast<HostBuffer*>(dev_dst_buf);
  bool was_already = (hb->host_ptr() == host_src_ptr);
  hb->re_alias(host_src_ptr, nbytes);
  dst_t.unsafeGetTensorImpl()->set_data(host_src_ptr);

  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  ET_LOG(Debug,
         "[mem] cpu: upload_from_host caller_ptr=%p bytes=%zu (%s)",
         host_src_ptr, nbytes,
         was_already ? "alias unchanged" : "re-aliased");
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error CpuInstance::download_to_host(
    ::executorch::runtime::EValue& dev_src_ev,
    Buffer* dev_src_buf,
    ::executorch::runtime::EValue& host_dst_ev,
    void* host_dst_ptr,
    QueueKind /*queue*/,
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  if (auto e = check_dependencies_(wait_for, signal);
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }
  if (!host_dst_ptr || !dev_src_buf || !dev_src_ev.isTensor() ||
      !host_dst_ev.isTensor()) {
    if (signal) signal->signal_failed(::executorch::runtime::Error::InvalidArgument);
    return ::executorch::runtime::Error::InvalidArgument;
  }

  auto& src_t = dev_src_ev.toTensor();
  auto& dst_t = host_dst_ev.toTensor();
  size_t nbytes = src_t.nbytes();

  if (auto e = ::executorch::runtime::resize_tensor(dst_t, src_t.sizes());
      e != ::executorch::runtime::Error::Ok) {
    if (signal) signal->signal_failed(e);
    return e;
  }

  // For CPU, "download to host" is the symmetric: re-alias the source
  // Buffer's pointer to the destination host pointer (caller's storage).
  // The kernel that produced dev_src_buf wrote directly into host_dst_ptr
  // already if alias was set up at bind_outputs time, in which case this
  // is idempotent.
  auto* hb = static_cast<HostBuffer*>(dev_src_buf);
  bool was_already = (hb->host_ptr() == host_dst_ptr);
  hb->re_alias(host_dst_ptr, nbytes);
  dst_t.unsafeGetTensorImpl()->set_data(host_dst_ptr);

  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  ET_LOG(Debug,
         "[mem] cpu: download_to_host caller_ptr=%p bytes=%zu (%s)",
         host_dst_ptr, nbytes,
         was_already ? "alias unchanged" : "re-aliased");
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error CpuInstance::check_dependencies_(
    ::executorch::runtime::Span<Event* const> wait_for, Event* signal) {
  return check_async_dependencies(wait_for, signal);
}

::executorch::runtime::Error CpuInstance::execute(
    CompiledSegment* segment,
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    BindingView bindings,
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  if (auto e = check_dependencies_(wait_for, signal);
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }

  auto* seg = static_cast<CpuCompiledSegment*>(segment);
  if (!seg) {
    return ::executorch::runtime::Error::InvalidArgument;
  }
  const auto* graph = seg->graph();
  if (!graph) return ::executorch::runtime::Error::InvalidState;

  // Verify each tensor EValue's TensorImpl::data_ptr matches the
  // currently-bound HostBuffer's host_ptr (after remap). The data_ptr
  // is established at prebind / bind_inputs / upload_from_host time;
  // by the time we get here it MUST match the bound Buffer.
  if (auto e = verify_segment_bindings(seg, graph, values, bindings,
                                        signal, "CpuInstance");
      e != ::executorch::runtime::Error::Ok) {
    return e;
  }

  // Drive ops via the existing portable kernel registry.
  ::executorch::runtime::KernelRuntimeContext kctx{};
  ::executorch::backends::portable::CpuGraph cpu_graph(kctx, values);

  for (uint32_t instr_idx : seg->instruction_indices()) {
    auto op = graph->get_instruction(instr_idx);
    const char* op_name = op.name();
    if (!op_name) {
      if (signal) signal->signal_failed(::executorch::runtime::Error::InvalidProgram);
      return ::executorch::runtime::Error::InvalidProgram;
    }

    ET_LOG(Debug, "CpuInstance: instr %u op='%s' (in=%zu, out=%zu)", instr_idx,
           op_name, op.num_inputs(), op.num_outputs());

    auto* handler =
        ::executorch::backends::portable::cpu_op_registry().try_get_op_fn(
            op_name);
    if (!handler) {
      ET_LOG(Error, "CpuInstance: no handler for %s", op_name);
      if (signal) signal->signal_failed(::executorch::runtime::Error::NotSupported);
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
    (*handler)(cpu_graph, args);

    if (kctx.failure_state() != ::executorch::runtime::Error::Ok) {
      auto err = kctx.failure_state();
      if (signal) signal->signal_failed(err);
      return err;
    }
  }

  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
  }
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Error CpuInstance::wait(Event* event) {
  if (!event) return ::executorch::runtime::Error::Ok;
  // CPU events are settled by the producing call. Just translate status.
  return wait_status(event->status(), event);
}

void CpuInstance::release_buffer(Buffer* buf) {
  // For Owned/NDM buffers held in owned_buffers_, the destructor of the
  // unique_ptr handles freeing on ~CpuInstance. release_buffer is called
  // for every owned buffer at ~Plan, which we treat as a no-op here
  // (the actual free happens at our destruction).
  //
  // For Aliasing buffers from the HostImportArena, this is called from
  // the executor at the top of the next execute() — also a no-op since
  // the arena reset() is what reclaims the slot.
  (void)buf;
}

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
