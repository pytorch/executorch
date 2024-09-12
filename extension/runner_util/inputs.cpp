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

using executorch::runtime::Error;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;

namespace executorch {
namespace extension {

Result<BufferCleanup> prepare_input_tensors(Method& method) {
  MethodMeta method_meta = method.method_meta();
  size_t num_inputs = method_meta.num_inputs();
  size_t num_allocated = 0;
  void** inputs = (void**)malloc(num_inputs * sizeof(void*));

  for (size_t i = 0; i < num_inputs; i++) {
    auto tag = method_meta.input_tag(i);
    if (!tag.ok()) {
      BufferCleanup cleanup({inputs, num_allocated});
      return tag.error();
    }
    if (tag.get() != Tag::Tensor) {
      ET_LOG(Debug, "Skipping non-tensor input %zu", i);
      continue;
    }
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
    if (!tensor_meta.ok()) {
      return tensor_meta.error();
    }
    // This input is a tensor. Allocate a buffer for it.
    void* data_ptr = malloc(tensor_meta->nbytes());
    inputs[num_allocated++] = data_ptr;

    // Create the tensor and set it as the input.
    Error err =
        internal::fill_and_set_input(method, tensor_meta.get(), i, data_ptr);
    if (err != Error::Ok) {
      ET_LOG(
          Error, "Failed to prepare input %zu: 0x%" PRIx32, i, (uint32_t)err);
      // The BufferCleanup will free the inputs when it goes out of scope.
      BufferCleanup cleanup({inputs, num_allocated});
      return err;
    }
  }
  return BufferCleanup({inputs, num_allocated});
}

} // namespace extension
} // namespace executorch
