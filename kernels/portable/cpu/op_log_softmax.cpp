/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <type_traits>

#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

Tensor& log_softmax_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_log_softmax_args(in, dim, half_to_float, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // Adjust for negative dim
  dim = dim < 0 ? dim + nonzero_dim(in) : dim;

  // For half-precision inputs, the exp-sum is accumulated in float to avoid
  // saturation (BFloat16 saturates near 256, Half near 2048). Matches ATen's
  // acc_type behavior. See also op_grid_sampler_2d.cpp.
  ET_SWITCH_FLOATHBF16_TYPES(
      in.scalar_type(), ctx, "_log_softmax.out", CTYPE, [&]() {
        using ACC = std::conditional_t<
            std::is_same_v<CTYPE, executorch::aten::Half> ||
                std::is_same_v<CTYPE, executorch::aten::BFloat16>,
            float,
            CTYPE>;
        const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
        CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

        apply_over_dim(
            [in_data, out_data](
                const size_t size, const size_t stride, const size_t base) {
              // calculate max in log_softmax dim. During log_softmax
              // computation each value is subtracted by the maximum in
              // value before calling exp to preserve numerical stability.
              const CTYPE max_in = apply_unary_reduce_fn(
                  [](const CTYPE val_in, CTYPE val_accum) {
                    return std::max(val_in, val_accum);
                  },
                  in_data + base,
                  size,
                  stride);

              const ACC exp_sum = apply_unary_map_reduce_fn<CTYPE, ACC>(
                  [max_in](const CTYPE val_in) {
                    return std::exp(
                        static_cast<ACC>(val_in) - static_cast<ACC>(max_in));
                  },
                  [](const ACC mapped_in, ACC val_accum) {
                    return val_accum + mapped_in;
                  },
                  in_data + base,
                  size,
                  stride);
              const ACC log_sum = std::log(exp_sum);

              apply_unary_map_fn(
                  [max_in, log_sum](const CTYPE val_in) {
                    return static_cast<CTYPE>(
                        static_cast<ACC>(val_in) - static_cast<ACC>(max_in) -
                        log_sum);
                  },
                  in_data + base,
                  out_data + base,
                  size,
                  stride);
            },
            in,
            dim);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
