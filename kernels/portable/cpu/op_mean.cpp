/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& mean_dim_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_mean_dim_args(in, dim_list, keepdim, dtype, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  std::optional<MapReduceOverDimListPlan> plan;
  if (in.numel() > 0) {
    plan.emplace(in, dim_list);
  }
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "mean.out";
  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
    ET_SWITCH_FLOATHBF16_TYPES(out.scalar_type(), ctx, op_name, CTYPE_OUT, [&] {
      CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
      const size_t num = get_reduced_dim_product(in, dim_list);
      const bool success = parallel_for_each_reduce_over_dim_list_output_index(
          in, dim_list, out, [&](const auto begin, const auto end) {
            for (const auto out_ix : c10::irange(begin, end)) {
              CTYPE_OUT sum = 0;
              if (plan.has_value()) {
                sum = plan->execute<CTYPE_IN, CTYPE_OUT>(
                    [](CTYPE_IN v) { return static_cast<CTYPE_OUT>(v); },
                    [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },
                    out_ix);
              }
              out_data[out_ix] = sum / static_cast<float>(num);
            }
          });
      ET_KERNEL_CHECK_MSG(ctx, success, Internal, , "parallel_for failed");
    });
  });

  return out;
}

Tensor& mean_dtype_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ScalarType> dtype,
    Tensor& out) {
  return mean_dim_out(ctx, in, ArrayRef<int64_t>(), false, dtype, out);
}

} // namespace native
} // namespace executor
} // namespace torch
