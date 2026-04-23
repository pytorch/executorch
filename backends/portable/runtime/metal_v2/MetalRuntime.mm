/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalRuntime.h"
#import "GpuStream.h"
#import "MetalOp.h"
#import "MetalOpRegistry.h"
#include <executorch/runtime/platform/log.h>
#include <cstring>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;
using runtime::EValue;
using runtime::Span;

MetalRuntime::MetalRuntime() : stream_(nullptr), graph_(nullptr) {
  stream_ = MetalStream::getDefault();
}

MetalRuntime::~MetalRuntime() {
  destroy();
}

bool MetalRuntime::is_available() const {
  return stream_ != nullptr;
}

bool MetalRuntime::has_op(const portable::OperatorCall& op, const portable::Graph& graph) const {
  const char* op_name = op.name();
  bool has = MetalOpRegistry::shared().hasOp(op_name);
  ET_LOG(Info, "MetalRuntime_v2: has_op('%s') = %d", op_name ? op_name : "null", has);
  return has;
}

Error MetalRuntime::init(
    const portable::Graph& graph,
    ArrayRef<portable::ExecutionSegment> segments) {

  graph_ = &graph;
  segments_.assign(segments.begin(), segments.end());

  ET_LOG(Info, "MetalRuntime_v2: initialized with %zu segments", segments_.size());
  return Error::Ok;
}

Error MetalRuntime::initialize_constants(Span<const uint32_t> value_indices) {
  // TODO: Copy constants to GPU
  // For now, constants are handled by the EValue system
  ET_LOG(Info, "MetalRuntime_v2: initialize_constants called (%zu values)", value_indices.size());
  return Error::Ok;
}

Error MetalRuntime::initialize_buffers(Span<const uint32_t> value_indices) {
  // Pre-allocate GPU buffers for the values we'll use
  // For unified memory, we just track which values we need
  for (auto idx : value_indices) {
    // Mark that this runtime handles this value
    // Actual allocation happens lazily or on first use
    valueBuffers_[idx] = nullptr;  // Will be set during upload
  }

  ET_LOG(Info, "MetalRuntime_v2: initialized %zu buffer slots", value_indices.size());
  return Error::Ok;
}

Error MetalRuntime::execute_segment(size_t segment_index, Span<EValue> values) {
  if (segment_index >= segments_.size()) {
    return Error::InvalidArgument;
  }

  const auto& segment = segments_[segment_index];

  for (auto instr_idx : segment.instruction_indices) {
    // Get instruction from graph
    auto op = graph_->get_instruction(instr_idx);
    const char* op_name = op.name();

    MetalOp* gpuOp = MetalOpRegistry::shared().get(op_name);
    if (!gpuOp) {
      ET_LOG(Error, "MetalRuntime_v2: unsupported op '%s'", op_name ? op_name : "null");
      return Error::NotSupported;
    }

    // Gather inputs and outputs
    std::vector<EValue*> inputs;
    std::vector<EValue*> outputs;

    for (size_t i = 0; i < op.num_inputs(); i++) {
      int32_t idx = op.input(i);
      if (idx >= 0 && static_cast<size_t>(idx) < values.size()) {
        inputs.push_back(&values[idx]);
      }
    }
    for (size_t i = 0; i < op.num_outputs(); i++) {
      int32_t idx = op.output(i);
      if (idx >= 0 && static_cast<size_t>(idx) < values.size()) {
        outputs.push_back(&values[idx]);
      }
    }

    // Ensure output tensors have allocated memory
    for (auto* output : outputs) {
      if (output->isTensor()) {
        auto& tensor = output->toTensor();
        if (!tensor.mutable_data_ptr() && tensor.nbytes() > 0) {
          // Allocate buffer for this output
          void* data = stream_->alloc(tensor.nbytes());
          if (!data) {
            ET_LOG(Error, "MetalRuntime_v2: failed to alloc %zu bytes for output", tensor.nbytes());
            return Error::MemoryAllocationFailed;
          }

          // Set the tensor's data pointer
          // Note: This requires mutable access to the tensor implementation
          // For ExecuTorch, we need to use the TensorImpl API
          auto impl = tensor.unsafeGetTensorImpl();
          if (impl) {
            impl->set_data(data);
            ET_LOG(Info, "MetalRuntime_v2: allocated %zu bytes for output at %p", tensor.nbytes(), data);
          }
        }
      }
    }

    // Dispatch - MetalStream handles replay automatically
    gpuOp->dispatch(
        stream_,
        MetalOp::EValuePtrSpan(inputs.data(), inputs.size()),
        MetalOp::EValuePtrSpan(outputs.data(), outputs.size()));
  }

  // Note: don't sync here - let dispatches accumulate
  // sync() happens in download_values()

  return Error::Ok;
}

Error MetalRuntime::upload_values(
    Span<const EValue> cpu_values,
    Span<const uint32_t> indices) {

  // For unified memory (Apple Silicon), no copy needed
  // The CPU and GPU share the same memory
  // Just track the buffers

  for (size_t i = 0; i < indices.size(); i++) {
    uint32_t idx = indices[i];
    if (idx < cpu_values.size() && cpu_values[idx].isTensor()) {
      auto& tensor = cpu_values[idx].toTensor();
      void* data = const_cast<void*>(tensor.const_data_ptr());
      if (data) {
        valueBuffers_[idx] = data;
        cachedSizes_[idx] = tensor.nbytes();
      }
    }
  }

  ET_LOG(Debug, "MetalRuntime_v2: uploaded %zu values (unified memory)", indices.size());
  return Error::Ok;
}

Error MetalRuntime::download_values(
    Span<EValue> cpu_values,
    Span<const uint32_t> indices) {

  // Sync GPU work first
  stream_->sync();

  // For unified memory, no copy needed
  // Data is already in CPU-accessible memory

  ET_LOG(Debug, "MetalRuntime_v2: downloaded %zu values (unified memory)", indices.size());
  return Error::Ok;
}

void MetalRuntime::destroy() {
  // Free any GPU buffers we allocated
  for (auto& [idx, buffer] : valueBuffers_) {
    if (buffer) {
      // Only free buffers we allocated via stream_->alloc()
      // Don't free buffers from unified memory (they come from EValue tensors)
    }
  }
  valueBuffers_.clear();
  cachedSizes_.clear();
  segments_.clear();
  graph_ = nullptr;

  ET_LOG(Info, "MetalRuntime_v2: destroyed");
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
