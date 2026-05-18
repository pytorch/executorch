/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <optional>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& sum_dim_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args(in, dim_list, keepdim, dtype, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  // Fast path: contiguous tensor, single innermost dim reduction, same dtype.
  // Bypasses generic MapReduceOverDimListPlan to use a tight vectorizable loop.
  if (in.numel() > 0 && dim_list.has_value() && dim_list.value().size() == 1 &&
      !executorch::runtime::isComplexType(in.scalar_type()) &&
      in.scalar_type() == out.scalar_type()) {
    const int64_t d = dim_list.value()[0] < 0 ? dim_list.value()[0] + in.dim()
                                              : dim_list.value()[0];
    if (d >= 0 && d < in.dim() && d == in.dim() - 1 &&
        tensor_is_contiguous(in)) {
      const int64_t reduce_size = in.size(d);
      const int64_t outer_size = in.numel() / reduce_size;

      // @lint-ignore CLANGTIDY facebook-hte-CArray
      static constexpr const char op_name[] = "sum.IntList_out";
      ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
        const CTYPE* in_data = in.const_data_ptr<CTYPE>();
        CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
        for (int64_t i = 0; i < outer_size; i++) {
          const CTYPE* row = in_data + i * reduce_size;
          CTYPE acc = 0;
          for (int64_t j = 0; j < reduce_size; j++) {
            acc += row[j];
          }
          out_data[i] = acc;
        }
      });
      return out;
    }
  }

  std::optional<MapReduceOverDimListPlan> plan;
  if (in.numel() > 0) {
    plan.emplace(in, dim_list);
  }
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "sum.IntList_out";

  if (executorch::runtime::isComplexType(in.scalar_type())) {
    ET_KERNEL_CHECK(
        ctx, in.scalar_type() == out.scalar_type(), InvalidArgument, out);

    ET_SWITCH_COMPLEXH_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
      CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
      const bool success = parallel_for_each_reduce_over_dim_list_output_index(
          in, dim_list, out, [&](const auto begin, const auto end) {
            for (const auto out_ix : c10::irange(begin, end)) {
              CTYPE sum(0, 0);
              if (plan.has_value()) {
                sum = plan->execute<CTYPE, CTYPE>(
                    [](CTYPE v) { return v; },
                    [](CTYPE outv, CTYPE acc) { return acc + outv; },
                    out_ix);
              }
              out_data[out_ix] = sum;
            }
          });
      ET_KERNEL_CHECK_MSG(ctx, success, Internal, , "parallel_for failed");
    });
  } else {
    ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
      ET_SWITCH_REALHBBF16_TYPES(
          out.scalar_type(), ctx, op_name, CTYPE_OUT, [&] {
            CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
            const bool success =
                parallel_for_each_reduce_over_dim_list_output_index(
                    in, dim_list, out, [&](const auto begin, const auto end) {
                      for (const auto out_ix : c10::irange(begin, end)) {
                        CTYPE_OUT sum = 0;
                        if (plan.has_value()) {
                          sum = plan->execute<CTYPE_IN, CTYPE_OUT>(
                              [](CTYPE_IN v) {
                                return static_cast<CTYPE_OUT>(v);
                              },
                              [](CTYPE_OUT outv, CTYPE_OUT acc) {
                                return acc + outv;
                              },
                              out_ix);
                        }
                        out_data[out_ix] = sum;
                      }
                    });
            ET_KERNEL_CHECK_MSG(
                ctx, success, Internal, , "parallel_for failed");
          });
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
