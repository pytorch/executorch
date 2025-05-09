/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <optional>

namespace cadence {
namespace impl {
namespace cpu {
namespace native {
namespace {
using ::executorch::runtime::getLeadingDims;

#define ET_FORALL_CADENCE_QUANTIZED_TYPES(_) \
  _(uint8_t, Byte)                           \
  _(int8_t, Char)

inline __attribute__((always_inline)) void linear_(
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const std::optional<::executorch::aten::Tensor>& bias,
    ::executorch::aten::Tensor& output) {
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

} // namespace
} // namespace native
} // namespace cpu
} // namespace impl
} // namespace cadence
