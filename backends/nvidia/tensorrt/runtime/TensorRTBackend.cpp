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
#include <executorch/runtime/platform/log.h>

#include <cuda_runtime.h> // @nolint

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
cudaStream_t g_shared_stream = nullptr; // NOLINT(facebook-avoid-non-const-global-variables)
int g_shared_stream_refcount = 0; // NOLINT(facebook-avoid-non-const-global-variables)

cudaStream_t get_or_create_shared_stream() { // NOLINT(facebook-hte-NullableReturn)
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

  MemoryAllocator* allocator =
      context.get_runtime_allocator();
  if (allocator == nullptr) {
    ET_LOG(Error, "Failed to get runtime allocator");
    return Error::InvalidState;
  }

  TensorRTExecutor* executor =
      allocator->allocateInstance<TensorRTExecutor>();
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
    bool uses_shared = !executor->owns_stream();
    executor->~TensorRTExecutor();
    if (uses_shared) {
      release_shared_stream();
    }
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

  if (num_inputs + num_outputs == 0) {
    ET_LOG(Error, "No inputs or outputs found");
    return Error::InvalidState;
  }

  std::vector<void*> input_buffers;
  std::vector<void*> output_buffers;
  input_buffers.reserve(num_inputs);
  output_buffers.reserve(num_outputs);

  size_t tensor_idx = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    EValue* arg = args[i];
    if (arg == nullptr || !arg->isTensor()) {
      continue;
    }

    ::executorch::aten::Tensor tensor = arg->toTensor();
    void* data_ptr = tensor.mutable_data_ptr();

    if (tensor_idx < num_inputs) {
      input_buffers.push_back(data_ptr);
    } else {
      output_buffers.push_back(data_ptr);
    }
    ++tensor_idx;
  }

  if (input_buffers.size() != num_inputs) {
    ET_LOG(
        Error,
        "Input buffer count mismatch: expected %zu, got %zu",
        num_inputs,
        input_buffers.size());
      return Error::InvalidArgument;
  }

  if (output_buffers.size() != num_outputs) {
    ET_LOG(
        Error,
        "Output buffer count mismatch: expected %zu, got %zu",
        num_outputs,
        output_buffers.size());
      return Error::InvalidArgument;
  }

  return executor->execute(
      input_buffers.data(),
      input_buffers.size(),
      output_buffers.data(),
      output_buffers.size());
}

void TensorRTBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    auto* executor = static_cast<TensorRTExecutor*>(handle);
    bool uses_shared = !executor->owns_stream();
    executor->~TensorRTExecutor();
    if (uses_shared) {
      release_shared_stream();
    }
  }
}

} // namespace tensorrt
} // namespace backends
} // namespace executorch

namespace {
executorch::backends::tensorrt::TensorRTBackend& get_backend() {
  static executorch::backends::tensorrt::TensorRTBackend backend;
  return backend;
}
const executorch::runtime::Backend backend_id{"TensorRTBackend", &get_backend()};
const auto registered = executorch::runtime::register_backend(backend_id);
} // namespace
