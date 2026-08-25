/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cmath>

#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& amax_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    ArrayRef<int64_t> dim_list,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_amin_amax_args(in, dim_list, keepdim, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "amax.out";

  // Fast path: contiguous tensor, single innermost dim reduction.
  // Bypasses the generic ReduceOverDimListPlan (per-element stride/index
  // recomputation via get_init_index)
  if (in.numel() > 0 && dim_list.size() == 1 &&
      !executorch::runtime::isComplexType(in.scalar_type()) &&
      tensor_is_default_dim_order(in)) {
    const int64_t d = dim_list[0] < 0 ? dim_list[0] + in.dim() : dim_list[0];
    if (d >= 0 && d < in.dim() && d == in.dim() - 1 &&
        tensor_is_contiguous(in)) {
      const int64_t reduce_size = in.size(d);
      const int64_t outer_size = in.numel() / reduce_size;

      ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&]() {
        const CTYPE* in_data = in.const_data_ptr<CTYPE>();
        CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
        for (int64_t i = 0; i < outer_size; i++) {
          const CTYPE* row = in_data + i * reduce_size;
          CTYPE max_v = row[0];
          for (int64_t j = 1; j < reduce_size; j++) {
            const CTYPE v = row[j];
            max_v = utils::isnan_override(v) || v > max_v ? v : max_v;
          }
          out_data[i] = max_v;
        }
      });
      return out;
    }
  }

  ReduceOverDimListPlan plan(in, dim_list);
  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&]() {
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    const bool success = parallel_for_each_reduce_over_dim_list_output_index(
        in, dim_list, out, [&](const auto begin, const auto end) {
          for (const auto out_ix : c10::irange(begin, end)) {
            out_data[out_ix] = plan.execute<CTYPE>(
                [](CTYPE v, CTYPE max_v) {
                  return utils::isnan_override(v) || v > max_v ? v : max_v;
                },
                out_ix);
          }
        });
    ET_KERNEL_CHECK_MSG(ctx, success, Internal, , "parallel_for failed");
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
