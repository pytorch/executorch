/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/arange_util.h>

namespace torch::executor::native {
#define ET_ARANGE_IMPL(ctx, start, numel, step, out, op_name)               \
  ET_SWITCH_REALHBF16_TYPES(out.scalar_type(), ctx, op_name, CTYPE, [&]() { \
    auto out_data = out.mutable_data_ptr<CTYPE>();                          \
    for (Tensor::SizesType i = 0; i < numel; ++i) {                         \
      out_data[i] = static_cast<CTYPE>(start + i * step);                   \
    }                                                                       \
  })

Tensor::SizesType
compute_arange_out_size(double start, double end, double step) {
  ET_CHECK_MSG(
      end > start, "end (%f) must be greater than start (%f)", end, start);
  ET_CHECK_MSG(step > 0, "step must be positive, got %f", step);
  Tensor::SizesType numel =
      static_cast<Tensor::SizesType>(std::ceil((end - start) / step));
  return numel;
}

void arange_out_impl(
    KernelRuntimeContext& ctx,
    double start,
    double end,
    double step,
    Tensor& out) {
  Tensor::SizesType numel = compute_arange_out_size(start, end, step);
  ET_ARANGE_IMPL(ctx, start, numel, step, out, "arange.start_out");
}

void arange_out_impl(KernelRuntimeContext& ctx, double end, Tensor& out) {
  ET_ARANGE_IMPL(ctx, 0.0, end, 1.0, out, "arange.out");
}

} // namespace torch::executor::native
