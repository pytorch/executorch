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
  const auto kSeq = input.size(1);
  const auto kH = input.size(2);
  const auto kHd = input.numel() / (kSeq * kH);
  for (int32_t s = 0; s < kSeq; ++s) {
    for (int32_t h = 0; h < kH; ++h) {
      for (int32_t hd_o = 0; hd_o < kHd / 2; ++hd_o) {
        float x_0 =
            input.const_data_ptr<float>()[s * kH * kHd + h * kHd + hd_o * 2];
        float x_1 =
            input
                .const_data_ptr<float>()[s * kH * kHd + h * kHd + hd_o * 2 + 1];
        int64_t token_id = s;
        if (pos.has_value()) {
          if (pos->scalar_type() == ::executorch::aten::ScalarType::Int) {
            token_id = pos.has_value() ? pos->const_data_ptr<int32_t>()[s] : s;
          } else {
            token_id = pos.has_value() ? pos->const_data_ptr<int64_t>()[s] : s;
          }
        }
        float sin =
            sin_tensor.const_data_ptr<float>()[token_id * kHd / 2 + hd_o];
        float cos =
            cos_tensor.const_data_ptr<float>()[token_id * kHd / 2 + hd_o];

        float out_0 = x_0 * cos - x_1 * sin;
        float out_1 = x_0 * sin + x_1 * cos;
        out.mutable_data_ptr<float>()[s * kH * kHd + h * kHd + hd_o * 2] =
            out_0;
        out.mutable_data_ptr<float>()[s * kH * kHd + h * kHd + hd_o * 2 + 1] =
            out_1;
      }
    }
  }

  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
