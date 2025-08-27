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

Tensor& amin_out(
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

  ReduceOverDimListPlan plan(in, dim_list);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "amin.out";

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&]() {
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    const bool success = parallel_for_each_reduce_over_dim_list_output_index(
        in, dim_list, out, [&](const auto begin, const auto end) {
          for (const auto out_ix : c10::irange(begin, end)) {
            out_data[out_ix] = plan.execute<CTYPE>(
                [](CTYPE v, CTYPE min_v) {
                  return utils::isnan_override(v) || v < min_v ? v : min_v;
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
