/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include "kernels.h"

namespace impl {
namespace HiFi {
namespace native {

// Note: this kernel assumes that the input and output share quantization
// parameters. If that is not the case, it will produce incorrect results.
template <typename T>
void quantized_relu_(
    const Tensor& input,
    const Tensor& in_zero_point,
    Tensor& output) {
  T q_zero_point = in_zero_point.const_data_ptr<T>()[0];
  const T* __restrict__ in = input.const_data_ptr<T>();
  T* __restrict__ out = output.mutable_data_ptr<T>();

  for (size_t i = 0, e = input.numel(); i < e; ++i) {
    out[i] = in[i] > q_zero_point ? in[i] : q_zero_point;
  }
}

void quantized_relu_out(
    const Tensor& input,
    const Tensor& in_zero_point,
    Tensor& output) {
  if (input.scalar_type() == exec_aten::ScalarType::Byte) {
    quantized_relu_<uint8_t>(input, in_zero_point, output);
  } else if (input.scalar_type() == exec_aten::ScalarType::Char) {
    quantized_relu_<int8_t>(input, in_zero_point, output);
  } else {
    ET_CHECK_MSG(false, "Unhandled input dtype %hhd", input.scalar_type());
  }
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
