/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/nvidia/tensorrt/runtime/TensorRTBackend.h>
#include <executorch/backends/nvidia/tensorrt/runtime/TensorRTBlobHeader.h>
#include <executorch/backends/nvidia/tensorrt/runtime/TensorRTExecutor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

#include <cuda_runtime.h>

namespace executorch {
namespace backends {
namespace tensorrt {

using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::register_backend;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

bool is_tensorrt_available() {
  return true;
}

// Shared CUDA stream for serialized execution across TRT delegates.
// When multiple TRT delegate subgraphs execute sequentially (the common case),
// sharing a single stream avoids synchronization overhead between subgraphs.
cudaStream_t g_shared_stream = nullptr;
int g_shared_stream_refcount = 0;

cudaStream_t get_or_create_shared_stream() {
  if (g_shared_stream == nullptr) {
    cudaError_t err = cudaStreamCreate(&g_shared_stream);
    if (err != cudaSuccess) {
      ET_LOG(Error, "Failed to create shared CUDA stream");
      return nullptr;
    }
  }
  ++g_shared_stream_refcount;
  return g_shared_stream;
}

void release_shared_stream() {
  if (g_shared_stream_refcount > 0) {
    --g_shared_stream_refcount;
  }
  if (g_shared_stream_refcount == 0 && g_shared_stream != nullptr) {
    cudaStreamDestroy(g_shared_stream);
    g_shared_stream = nullptr;
  }
}

} // namespace

bool TensorRTBackend::is_available() const {
  return is_tensorrt_available();
}

Result<DelegateHandle*> TensorRTBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  (void)compile_specs;

  if (!is_available()) {
    ET_LOG(Error, "TensorRT backend is not available");
    return Error::NotSupported;
  }

  if (processed == nullptr || processed->data() == nullptr) {
    ET_LOG(Error, "Invalid processed buffer");
    return Error::InvalidArgument;
  }

  const void* blob_data = processed->data();
  const size_t blob_size = processed->size();

  TensorRTBlobHeader header{};
  if (!parse_blob_header(blob_data, blob_size, header)) {
    ET_LOG(Error, "Failed to parse TensorRT blob header");
    return Error::InvalidArgument;
  }

  MemoryAllocator* allocator = context.get_runtime_allocator();
  if (allocator == nullptr) {
    ET_LOG(Error, "Failed to get runtime allocator");
    return Error::InvalidState;
  }

  TensorRTExecutor* executor = allocator->allocateInstance<TensorRTExecutor>();
  if (executor == nullptr) {
    ET_LOG(Error, "Failed to allocate TensorRT executor");
    return Error::MemoryAllocationFailed;
  }

  new (executor) TensorRTExecutor();

  // Share a CUDA stream across all TRT delegate instances for serialized
  // execution. This avoids synchronization overhead between subgraphs when
  // they execute sequentially (the common case).
  cudaStream_t shared = get_or_create_shared_stream();
  if (shared != nullptr) {
    executor->set_cuda_stream(shared, false);
  }

  Error err = executor->initialize(blob_data, blob_size);
  if (err != Error::Ok) {
    ET_LOG(Error, "Failed to initialize TensorRT executor");
    executor->~TensorRTExecutor();
    return err;
  }

  processed->Free();

  return static_cast<DelegateHandle*>(executor);
}

Error TensorRTBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  (void)context;

  if (handle == nullptr) {
    ET_LOG(Error, "Invalid delegate handle");
    return Error::InvalidArgument;
  }

  auto* executor = static_cast<TensorRTExecutor*>(handle);

  if (!executor->is_initialized()) {
    ET_LOG(Error, "Executor not initialized");
    return Error::InvalidState;
  }

  size_t num_inputs = executor->get_num_inputs();
  size_t num_outputs = executor->get_num_outputs();

  // ExecuTorch passes [inputs..., outputs...] in args (same as CUDA backend).
  if (args.size() < num_inputs + num_outputs) {
    ET_LOG(
        Error,
        "args size %zu < inputs %zu + outputs %zu",
        args.size(),
        num_inputs,
        num_outputs);
    return Error::InvalidArgument;
  }

  // Extract input pointers and shapes.
  std::vector<void*> input_buffers(num_inputs);
  std::vector<std::vector<int64_t>> input_shapes(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    auto& tensor = args[i]->toTensor();
    input_buffers[i] = tensor.mutable_data_ptr();
    auto sizes = tensor.sizes();
    input_shapes[i].assign(sizes.begin(), sizes.end());
  }

  // Extract output pointers.
  std::vector<void*> output_buffers(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    output_buffers[i] = args[i + num_inputs]->toTensor().mutable_data_ptr();
  }

  auto err = executor->execute(
      input_buffers.data(),
      input_shapes,
      num_inputs,
      output_buffers.data(),
      num_outputs);

  if (err != Error::Ok) {
    return err;
  }

  // For dynamic shapes, resize output tensors to match TRT-inferred shapes.
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto& shape = executor->get_output_shape(i);
    if (!shape.empty()) {
      auto& tensor = args[i + num_inputs]->toTensor();
      auto current = tensor.sizes();
      bool needs_resize = (static_cast<size_t>(current.size()) != shape.size());
      if (!needs_resize) {
        for (size_t d = 0; d < shape.size(); ++d) {
          if (current[d] != shape[d]) {
            needs_resize = true;
            break;
          }
        }
      }
      if (needs_resize) {
        std::vector<::executorch::aten::SizesType> new_sizes(
            shape.begin(), shape.end());
        auto resize_err = executorch::runtime::resize_tensor(
            tensor, {new_sizes.data(), new_sizes.size()});
        if (resize_err != Error::Ok) {
          ET_LOG(Error, "Failed to resize output tensor %zu", i);
        }
      }
    }
  }

  return Error::Ok;
}

void TensorRTBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    auto* executor = static_cast<TensorRTExecutor*>(handle);
    executor->~TensorRTExecutor();
    release_shared_stream();
  }
}

} // namespace tensorrt
} // namespace backends
} // namespace executorch

namespace {
auto backend = executorch::backends::tensorrt::TensorRTBackend();
executorch::runtime::Backend backend_id{"TensorRTBackend", &backend};
static auto registered = executorch::runtime::register_backend(backend_id);
} // namespace
