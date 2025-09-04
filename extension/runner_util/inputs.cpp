/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/runner_util/inputs.h>

#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/log.h>

using executorch::ET_RUNTIME_NAMESPACE::Method;
using executorch::ET_RUNTIME_NAMESPACE::MethodMeta;
using executorch::ET_RUNTIME_NAMESPACE::TensorInfo;
using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::Tag;

namespace executorch {
namespace extension {

Result<BufferCleanup> prepare_input_tensors(
    Method& method,
    PrepareInputTensorsOptions options) {
  MethodMeta method_meta = method.method_meta();
  size_t num_inputs = method_meta.num_inputs();

  // A large number of small allocations could exhaust the heap even if the
  // total size is smaller than the limit.
  ET_CHECK_OR_RETURN_ERROR(
      num_inputs <= options.max_inputs,
      InvalidProgram,
      "Too many inputs: %zu > %zu",
      num_inputs,
      options.max_inputs);

  // Allocate memory for the inputs array
  void** inputs = (void**)malloc(num_inputs * sizeof(void*));
  ET_CHECK_OR_RETURN_ERROR(
      inputs != nullptr,
      MemoryAllocationFailed,
      "malloc(%zd) failed",
      num_inputs * sizeof(void*));

  // Allocate memory for each input tensor.
  size_t total_size = 0;
  size_t num_allocated = 0;
  for (size_t i = 0; i < num_inputs; i++) {
    auto tag = method_meta.input_tag(i);
    if (!tag.ok()) {
      // The BufferCleanup will free the inputs when it goes out of scope.
      BufferCleanup cleanup({inputs, num_allocated});
      return tag.error();
    }
    if (tag.get() == Tag::None) {
      Error err = method.set_input(runtime::EValue(), i);
      if (err != Error::Ok) {
        BufferCleanup cleanup({inputs, num_allocated});
        return err;
      }
      continue;
    }
    if (tag.get() != Tag::Tensor) {
      ET_LOG(Debug, "Skipping non-tensor input %zu", i);
      continue;
    }
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
    if (!tensor_meta.ok()) {
      BufferCleanup cleanup({inputs, num_allocated});
      return tensor_meta.error();
    }
    // This input is a tensor. Allocate a buffer for it.
    size_t tensor_size = tensor_meta->nbytes();
    total_size += tensor_size;
    if (total_size > options.max_total_allocation_size) {
      ET_LOG(
          Error,
          "Allocating %zu bytes for input %zu would exceed "
          "max_total_allocation_size %zu",
          tensor_size,
          i,
          options.max_total_allocation_size);
      BufferCleanup cleanup({inputs, num_allocated});
      return Error::InvalidProgram;
    }
    void* data_ptr = malloc(tensor_size);
    if (data_ptr == nullptr) {
      ET_LOG(Error, "malloc(%zu) failed for input %zu", tensor_size, i);
      BufferCleanup cleanup({inputs, num_allocated});
      return Error::MemoryAllocationFailed;
    }
    inputs[num_allocated++] = data_ptr;

    // Create the tensor and set it as the input.
    Error err =
        internal::fill_and_set_input(method, tensor_meta.get(), i, data_ptr);
    if (err != Error::Ok) {
      ET_LOG(
          Error, "Failed to prepare input %zu: 0x%" PRIx32, i, (uint32_t)err);
      BufferCleanup cleanup({inputs, num_allocated});
      return err;
    }
  }

  return BufferCleanup({inputs, num_allocated});
}

} // namespace extension
} // namespace executorch
