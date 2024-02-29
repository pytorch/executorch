/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNExecutor.h>

namespace torch {
namespace executor {
namespace xnnpack {
namespace delegate {

Error XNNExecutor::set_external_input(
    uint32_t id,
    Tensor* input,
    struct XNNShape* shape) {
  // TODO(T165403530): Test ensure accuracy for int64 --> float32 conversion
  if (input->scalar_type() == ScalarType::Long) {
    // Input data type is int64. However, XNNPACK doesn't support
    // int64. This means that the data needs to be casted to float
    // In order for XNNPACK to properly use it.
    const int64_t* data_64 = input->const_data_ptr<int64_t>();
    float* data_f32 = input->mutable_data_ptr<float>();
    for (int j = 0; j < input->numel(); j++) {
      data_f32[j] = data_64[j];
    }
  }
  if (input->dim() != shape->num_dims) {
    ET_LOG(Error, "Input dim mismatch between tensor and shape struct");
  }

#ifdef ENABLE_DYNAMIC_QUANTIZATION
  externals_.emplace_back(xnn_external_value{
      id,
      input->mutable_data_ptr(),
      static_cast<size_t>(shape->num_dims),
      shape->dim});
#else
  externals_.emplace_back(xnn_external_value{id, input->mutable_data_ptr()});
#endif
  return Error::Ok;
}

} // namespace delegate
} // namespace xnnpack
} // namespace executor
} // namespace torch
