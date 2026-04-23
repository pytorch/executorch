/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/portable/runtime_v2/metal/MetalInstance.h>

#include <executorch/backends/portable/runtime_v2/api/InstanceUtils.h>
#include <executorch/backends/portable/runtime_v2/metal/MetalBuffer.h>
#include <executorch/backends/portable/runtime_v2/metal/MetalEvent.h>
#include <executorch/backends/portable/runtime_v2/metal/MetalProvider.h>

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
namespace portable_v2 {

namespace metal_v2_ns = ::executorch::backends::metal_v2;
namespace runtime = ::executorch::runtime;

MetalInstance::MetalInstance(MetalProvider* provider, InstanceId id)
    : provider_(provider),
      stream_(provider ? provider->stream() : nullptr),
      id_(id) {}

MetalInstance::~MetalInstance() {
  // Drain any in-flight work before tearing down owned buffers, so the
  // GPU isn't still referencing them.
  drain();
  // Buffers are owned via unique_ptr vector; destructors handle pool
  // returns (Owned mode) or are no-ops (Aliasing/NdmAlias).
  owned_buffers_.clear();
}

runtime::Error MetalInstance::check_dependencies_(
    runtime::Span<Event* const> wait_for, Event* signal) {
  for (Event* e : wait_for) {
    if (!e) continue;
    if (auto err = wait(e); err != runtime::Error::Ok) {
      if (signal) signal->signal_poisoned(err);
      return runtime::Error::Internal;  // AsyncDependencyFailed
    }
  }
  return runtime::Error::Ok;
}

runtime::Result<CompiledSegment*> MetalInstance::compile_segment(
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

runtime::Error MetalInstance::allocate_all(
    runtime::Span<const AllocRequest> requests,
    runtime::Span<const runtime::EValue> values,
    runtime::Span<Buffer*> out_buffers) {
  if (!stream_) return runtime::Error::InvalidState;
  if (requests.size() != out_buffers.size()) {
    return runtime::Error::InvalidArgument;
  }
  Location loc = Location::on(provider_->id());
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& req = requests[i];
    if (req.value_id >= values.size() ||
        !values[req.value_id].isTensor()) {
      ET_LOG(Error,
             "MetalInstance::allocate_all: value_id=%u missing or not a tensor",
             req.value_id);
      return runtime::Error::InvalidArgument;
    }
    const auto& tensor = values[req.value_id].toTensor();
    size_t nbytes = tensor.nbytes();

    // Cross-runtime synthetic with a host-addressable source: try
    // zero-copy alias. Register the source's host pointer with the
    // stream as a standalone MTLBuffer (newBufferWithBytesNoCopy). If
    // Metal accepts, return a Mode::Aliasing MetalBuffer wrapping the
    // host pointer — NO pool allocation.
    if (req.host_alias && req.host_alias->host_ptr()) {
      void* p = req.host_alias->host_ptr();
      if (stream_->registerExternalBuffer(p, nbytes, /*strict_zero_copy=*/true)) {
        auto* buf = MetalBuffer::alias(stream_, loc, p, nbytes);
        out_buffers[i] = buf;
        ET_LOG(Debug,
               "[mem] metal: allocate_all value_id=%u bytes=%zu host_alias=%p (zero-copy alias)",
               req.value_id, nbytes, p);
        owned_buffers_.emplace_back(buf);
        continue;
      }
      // Metal refused zero-copy on this pointer (alignment etc.). Fall
      // through to a fresh pool allocation; per-execute upload_from_host
      // will memcpy bytes in.
    }

    // Fresh pool allocation: intermediates, IO destinations, alias-refused
    // synthetics, and (in the future) alias hints whose source has no
    // host_ptr (Vulkan-produced values).
    void* ptr = stream_->alloc(nbytes);
    if (!ptr) return runtime::Error::MemoryAllocationFailed;
    (void)stream_->bufferForPtr(ptr, nbytes);
    auto* buf = MetalBuffer::allocate(stream_, loc, ptr, nbytes);
    out_buffers[i] = buf;
    ET_LOG(Debug,
           "[mem] metal: allocate_all value_id=%u bytes=%zu host_ptr=%p",
           req.value_id, nbytes, ptr);
    owned_buffers_.emplace_back(buf);
  }
  return runtime::Error::Ok;
}

runtime::Result<Buffer*> MetalInstance::upload_constant(
    const runtime::NamedDataMap& ndm, std::string_view key) {
  if (!stream_) return runtime::Error::InvalidState;
  auto fb_result = ndm.get_data(key);
  if (!fb_result.ok()) {
    ET_LOG(Error, "MetalInstance: upload_constant: NDM key '%.*s' not found",
           static_cast<int>(key.size()), key.data());
    return fb_result.error();
  }
  runtime::FreeableBuffer fb = std::move(fb_result.get());
  void* ptr = const_cast<void*>(fb.data());
  size_t bytes = fb.size();
  // Zero-copy alias: register the FreeableBuffer's mmap'd region with
  // MetalStream so kernels' bufferForPtr() resolves to a wrapping
  // MTLBuffer (newBufferWithBytesNoCopy under the hood).
  stream_->registerExternalBuffer(ptr, bytes);
  (void)stream_->bufferForPtr(ptr, bytes);
  Location loc = Location::on(provider_->id());
  auto* buf = MetalBuffer::alias_ndm(stream_, loc, std::move(fb));
  owned_buffers_.emplace_back(buf);
  ET_LOG(Debug,
         "[mem] metal: upload_constant key='%.*s' bytes=%zu host_ptr=%p (zero-copy NDM alias via registerExternalBuffer)",
         static_cast<int>(key.size()), key.data(), bytes, ptr);
  return buf;
}

std::unique_ptr<Event> MetalInstance::make_event() {
  return std::make_unique<MetalEvent>();
}

runtime::Error MetalInstance::upload_from_host(
    runtime::EValue& host_src_ev,
    void* host_src_ptr,
    runtime::EValue& dev_dst_ev,
    Buffer* dev_dst_buf,
    QueueKind /*queue*/,
    runtime::Span<Event* const> wait_for,
    Event* signal) {
  // 1. Wait on producer event(s).
  if (auto e = check_dependencies_(wait_for, signal);
      e != runtime::Error::Ok) {
    return e;
  }

  // 2. Validate.
  if (!host_src_ptr || !dev_dst_buf || !host_src_ev.isTensor() ||
      !dev_dst_ev.isTensor()) {
    if (signal) signal->signal_failed(runtime::Error::InvalidArgument);
    return runtime::Error::InvalidArgument;
  }

  auto& src_t = host_src_ev.toTensor();
  auto& dst_t = dev_dst_ev.toTensor();
  size_t nbytes = src_t.nbytes();

  // 3. Propagate shape.
  if (auto e = runtime::resize_tensor(dst_t, src_t.sizes());
      e != runtime::Error::Ok) {
    if (signal) signal->signal_failed(e);
    return e;
  }

  auto* mb = static_cast<MetalBuffer*>(dev_dst_buf);

  // 4. Skip-if-same: if dest is already aliased here, no work.
  if (mb->host_ptr() == host_src_ptr) {
    dst_t.unsafeGetTensorImpl()->set_data(host_src_ptr);
    if (signal) {
      signal->prepare_signal();
      signal->signal_complete();
      }
    ET_LOG(Debug,
           "[mem] metal: upload_from_host caller_ptr=%p bytes=%zu (alias unchanged)",
           host_src_ptr, nbytes);
    return runtime::Error::Ok;
  }

  // 5. Try zero-copy: register caller pointer with the stream as a
  // standalone MTLBuffer (newBufferWithBytesNoCopy). If Metal accepts,
  // re-alias the destination MetalBuffer to point at host_src_ptr (the
  // pool-allocated storage is returned to the pool).
  if (stream_->registerExternalBuffer(
          host_src_ptr, nbytes, /*strict_zero_copy=*/true)) {
    mb->re_alias(host_src_ptr, nbytes);
    dst_t.unsafeGetTensorImpl()->set_data(host_src_ptr);
    if (signal) {
      signal->prepare_signal();
      signal->signal_complete();
      }
    ET_LOG(Debug,
           "[mem] metal: upload_from_host caller_ptr=%p bytes=%zu (re-aliased zero-copy)",
           host_src_ptr, nbytes);
    return runtime::Error::Ok;
  }

  // 6. Fallback: Metal refused zero-copy on this pointer. memcpy into
  // the existing MetalBuffer's pool-allocated storage.
  void* dst_ptr = mb->host_ptr();
  if (!dst_ptr) {
    if (signal) signal->signal_failed(runtime::Error::InvalidArgument);
    return runtime::Error::InvalidArgument;
  }
  std::memcpy(dst_ptr, host_src_ptr, nbytes);
  dst_t.unsafeGetTensorImpl()->set_data(dst_ptr);
  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
    }
  ET_LOG(Debug,
         "[mem] metal: upload_from_host caller_ptr=%p bytes=%zu (memcpy fallback; zero-copy refused)",
         host_src_ptr, nbytes);
  return runtime::Error::Ok;
}

runtime::Error MetalInstance::download_to_host(
    runtime::EValue& dev_src_ev,
    Buffer* dev_src_buf,
    runtime::EValue& host_dst_ev,
    void* host_dst_ptr,
    QueueKind /*queue*/,
    runtime::Span<Event* const> wait_for,
    Event* signal) {
  // 1. Wait on producer event(s).
  if (auto e = check_dependencies_(wait_for, signal);
      e != runtime::Error::Ok) {
    return e;
  }

  // 2. Validate.
  if (!host_dst_ptr || !dev_src_buf || !dev_src_ev.isTensor() ||
      !host_dst_ev.isTensor()) {
    if (signal) signal->signal_failed(runtime::Error::InvalidArgument);
    return runtime::Error::InvalidArgument;
  }

  auto& src_t = dev_src_ev.toTensor();
  auto& dst_t = host_dst_ev.toTensor();
  size_t nbytes = src_t.nbytes();

  // 3. Propagate shape.
  if (auto e = runtime::resize_tensor(dst_t, src_t.sizes());
      e != runtime::Error::Ok) {
    if (signal) signal->signal_failed(e);
    return e;
  }

  auto* mb = static_cast<MetalBuffer*>(dev_src_buf);

  // 4. Sync GPU work so any pending writes to dev_src_buf are visible.
  stream_->sync();

  // 5. Skip-if-same: dev_src_buf already aliases host_dst_ptr (kernel
  // wrote directly into caller storage).
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

  // 6. Try zero-copy alias (uncommon for outputs but symmetric).
  if (stream_->registerExternalBuffer(
          host_dst_ptr, nbytes, /*strict_zero_copy=*/true)) {
    // Need to memcpy current bytes since the buffer just got rebound.
    void* old_ptr = mb->host_ptr();
    std::memcpy(host_dst_ptr, old_ptr, nbytes);
    mb->re_alias(host_dst_ptr, nbytes);
    if (signal) {
      signal->prepare_signal();
      signal->signal_complete();
      }
    ET_LOG(Debug,
           "[mem] metal: download_to_host caller_ptr=%p bytes=%zu (re-aliased after copy)",
           host_dst_ptr, nbytes);
    return runtime::Error::Ok;
  }

  // 7. Fallback: memcpy from device buffer to caller storage.
  void* src_ptr = mb->host_ptr();
  if (!src_ptr) {
    if (signal) signal->signal_failed(runtime::Error::InvalidArgument);
    return runtime::Error::InvalidArgument;
  }
  std::memcpy(host_dst_ptr, src_ptr, nbytes);
  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
    }
  ET_LOG(Debug,
         "[mem] metal: download_to_host caller_ptr=%p bytes=%zu (memcpy fallback)",
         host_dst_ptr, nbytes);
  return runtime::Error::Ok;
}

runtime::Error MetalInstance::execute(
    CompiledSegment* segment,
    runtime::Span<runtime::EValue> values,
    BindingView bindings,
    runtime::Span<Event* const> wait_for,
    Event* signal) {
  if (auto e = check_dependencies_(wait_for, signal);
      e != runtime::Error::Ok) {
    return e;
  }

  auto* seg = static_cast<MetalCompiledSegment*>(segment);
  if (!seg) return runtime::Error::InvalidArgument;
  const auto* graph = seg->graph();
  if (!graph) return runtime::Error::InvalidState;

  // Verify each tensor EValue's TensorImpl::data_ptr matches the
  // currently-bound MetalBuffer's host_ptr (= [mtlBuffer contents])
  // after remap. The data_ptr is established at prebind / bind_inputs /
  // upload_from_host time; by the time we get here it MUST match the
  // bound Buffer.
  if (auto e = verify_segment_bindings(seg, graph, values, bindings,
                                        signal, "MetalInstance");
      e != runtime::Error::Ok) {
    return e;
  }

  // Dispatch each instruction via the metal_v2 registry.
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
      ET_LOG(Error, "MetalInstance: op '%s' not in MetalOpRegistry", op_name);
      if (signal) signal->signal_failed(runtime::Error::NotSupported);
      return runtime::Error::NotSupported;
    }

    ET_LOG(Debug, "MetalInstance: instr %u op='%s' (in=%zu, out=%zu)",
           instr_idx, op_name, op.num_inputs(), op.num_outputs());

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

  // Flush + wait so the shape-on-event contract holds: by the time
  // signal goes Complete, host can read both shape (already on
  // TensorImpls — set host-side by each op's resizeOutput before
  // dispatch) AND bytes (ensured by stream->wait()).
  stream_->sync();

  if (signal) {
    signal->prepare_signal();
    signal->signal_complete();
    }
  return runtime::Error::Ok;
}

runtime::Error MetalInstance::wait(Event* event) {
  if (!event) return runtime::Error::Ok;
  // Spin until the event's atomic status reaches a terminal state.
  // For execute() and transfer_tensor(), we synchronously sync the
  // stream before signaling, so by the time the executor reaches a
  // wait() the event is already Complete and this is a few-load
  // hot path.
  while (true) {
    auto s = event->status();
    if (s == EventStatus::Complete) return runtime::Error::Ok;
    if (s == EventStatus::Failed || s == EventStatus::Poisoned) {
      return event->error();
    }
    // Pending: yield. (Acceptable for v1; a condvar is the future
    // optimization if we move signaling into completion handlers.)
  }
}

void MetalInstance::drain() {
  if (stream_) stream_->sync();
}

void MetalInstance::release_buffer(Buffer* buf) {
  // All Buffers live in owned_buffers_; their unique_ptrs run their
  // destructors at ~MetalInstance, which return Owned-mode pool memory
  // and release NdmAlias FreeableBuffers. Aliasing-mode Buffers (those
  // re-aliased via upload_from_host) are no-op destructors.
  // Per-Plan release_buffer is a no-op; lifetime is tied to the Instance.
  (void)buf;
}

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
