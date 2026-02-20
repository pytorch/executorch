/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_fully_connected.h>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::optional;
using ::executorch::aten::Tensor;
using ::executorch::runtime::getLeadingDims;
using ::executorch::runtime::KernelRuntimeContext;

void linear(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias,
    Tensor& output) {
  const float* __restrict__ input_data = input.const_data_ptr<float>();
  const float* __restrict__ weight_data = weight.const_data_ptr<float>();
  const float* __restrict__ bias_data = bias.value().const_data_ptr<float>();
  float* __restrict__ output_data = output.mutable_data_ptr<float>();

  // input comes in shape [batch_size, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [batch_size, out_dim]
  // Perform matrix multiply (M x N) x (N x P) => M x P
  int64_t M = weight.size(0); // = out_dim
  int64_t N = weight.size(1); // = in_dim

  // Given an N-dimensional input [d0, d1, d2, ..., d_{N-2}, d_{N-1}], the
  // leading dimensions is d0 * d1 * ... * d_{N-2}
  int64_t leading_dims = getLeadingDims(input, input.dim() - 1);

  for (int i = 0; i < leading_dims; ++i) {
    for (int j = 0; j < M; ++j) {
      float sum = bias_data[j];
      for (int k = 0; k < N; ++k) {
        sum += input_data[i * N + k] * weight_data[j * N + k];
      }
      output_data[i * M + j] = sum;
    }
  }
}

Tensor& fully_connected_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias,
    Tensor& output) {
  linear(input, weight, bias, output);
  return output;
}

} // namespace native
} // namespace generic
} // namespace impl
