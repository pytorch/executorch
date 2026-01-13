/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "executorch/backends/cadence/generic/operators/op_rope.h"

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::optional;
using ::executorch::aten::Tensor;

Tensor& rope_out(
    ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& sin_tensor,
    const Tensor& cos_tensor,
    const optional<Tensor>& pos,
    Tensor& out) {
  // Input shape is [1, seq, h, hd / 2, 2] or [1, seq, h, hd]
  const ssize_t seq_length = input.size(1);
  const ssize_t num_heads = input.size(2);
  const ssize_t head_dimension = input.numel() / (seq_length * num_heads);
  const ssize_t head_dimension_by_two = head_dimension / 2;
  for (int32_t s = 0; s < seq_length; ++s) {
    for (int32_t h = 0; h < num_heads; ++h) {
      for (int32_t hd_o = 0; hd_o < head_dimension_by_two; ++hd_o) {
        // Process 2 elements in head dimension at a time.
        const float x_0 = input.const_data_ptr<float>()
                              [s * num_heads * head_dimension +
                               h * head_dimension + hd_o * 2];
        const float x_1 = input.const_data_ptr<float>()
                              [s * num_heads * head_dimension +
                               h * head_dimension + hd_o * 2 + 1];
        int64_t token_id = s;
        if (pos.has_value()) {
          if (pos->scalar_type() == ::executorch::aten::ScalarType::Int) {
            token_id = pos.has_value() ? pos->const_data_ptr<int32_t>()[s] : s;
          } else {
            token_id = pos.has_value() ? pos->const_data_ptr<int64_t>()[s] : s;
          }
        }

        const float sin = sin_tensor.const_data_ptr<
            float>()[token_id * head_dimension_by_two + hd_o];
        const float cos = cos_tensor.const_data_ptr<
            float>()[token_id * head_dimension_by_two + hd_o];

        const float out_0 = x_0 * cos - x_1 * sin;
        out.mutable_data_ptr<float>()
            [s * num_heads * head_dimension + h * head_dimension + hd_o * 2] =
            out_0;

        const float out_1 = x_0 * sin + x_1 * cos;
        out.mutable_data_ptr<float>()
            [s * num_heads * head_dimension + h * head_dimension + hd_o * 2 +
             1] = out_1;
      }
    }
  }

  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
