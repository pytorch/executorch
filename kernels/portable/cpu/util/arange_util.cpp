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
    for (executorch::aten::SizesType i = 0; i < numel; ++i) {               \
      out_data[i] = static_cast<CTYPE>(start + i * step);                   \
    }                                                                       \
  })

executorch::aten::SizesType
compute_arange_out_size(double start, double end, double step) {
  executorch::aten::SizesType numel =
      static_cast<executorch::aten::SizesType>(std::ceil((end - start) / step));

  ET_CHECK_MSG(
      numel >= 0,
      "numel should be non-negative, but got (%" PRId64
      "). start (%f), end (%f), step (%f)",
      static_cast<int64_t>(numel),
      start,
      end,
      step);
  return numel;
}

void arange_out_impl(
    KernelRuntimeContext& ctx,
    double start,
    double end,
    double step,
    Tensor& out) {
  (void)ctx;
  executorch::aten::SizesType numel = compute_arange_out_size(start, end, step);
  ET_ARANGE_IMPL(ctx, start, numel, step, out, "arange.start_out");
}

void arange_out_impl(KernelRuntimeContext& ctx, double end, Tensor& out) {
  (void)ctx;
  ET_ARANGE_IMPL(ctx, 0.0, end, 1.0, out, "arange.out");
}

} // namespace torch::executor::native
