/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cmath>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace {

template <typename CTYPE_IN, typename CTYPE_OUT>
void compute_var_mean(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& var_out,
    Tensor& mean_out,
    optional<ArrayRef<int64_t>> dim_list,
    const size_t num,
    const double denominator) {
  CTYPE_OUT* var_data = var_out.mutable_data_ptr<CTYPE_OUT>();
  CTYPE_OUT* mean_data = mean_out.mutable_data_ptr<CTYPE_OUT>();
  if (num == 0 || denominator <= 0) {
    for (const auto out_ix : c10::irange(var_out.numel())) {
      var_data[out_ix] = NAN;
      mean_data[out_ix] = NAN;
    }
  } else if (in.numel() > 0) {
    // Fast path: contiguous tensor, single innermost dim reduction, same dtype.
    bool used_fast_path = false;
    if (dim_list.has_value() && dim_list.value().size() == 1 &&
        in.scalar_type() == var_out.scalar_type()) {
      const int64_t d = dim_list.value()[0] < 0 ? dim_list.value()[0] + in.dim()
                                                : dim_list.value()[0];
      if (d >= 0 && d < in.dim() && d == in.dim() - 1 &&
          tensor_is_contiguous(in)) {
        used_fast_path = true;
        const int64_t reduce_size = in.size(d);
        const int64_t outer_size = in.numel() / reduce_size;
        const CTYPE_OUT cnum = static_cast<CTYPE_OUT>(num);
        const CTYPE_OUT cdenom = static_cast<CTYPE_OUT>(denominator);
        const CTYPE_IN* in_data = in.const_data_ptr<CTYPE_IN>();
        for (int64_t i = 0; i < outer_size; i++) {
          const CTYPE_IN* row = in_data + i * reduce_size;
          // Pass 1: compute mean
          CTYPE_OUT sum = 0;
          for (int64_t j = 0; j < reduce_size; j++) {
            sum += static_cast<CTYPE_OUT>(row[j]);
          }
          CTYPE_OUT mean = sum / cnum;
          mean_data[i] = mean;
          // Pass 2: compute variance
          CTYPE_OUT sum2 = 0;
          for (int64_t j = 0; j < reduce_size; j++) {
            CTYPE_OUT diff = static_cast<CTYPE_OUT>(row[j]) - mean;
            sum2 += diff * diff;
          }
          var_data[i] = sum2 / cdenom;
        }
      }
    }
    if (!used_fast_path) {
      MapReduceOverDimListPlan plan(in, dim_list);
      const bool success = parallel_for_each_reduce_over_dim_list_output_index(
          in, dim_list, var_out, [&](const auto begin, const auto end) {
            for (const auto out_ix : c10::irange(begin, end)) {
              // Pass 1: compute sum -> mean
              CTYPE_OUT sum = plan.execute<CTYPE_IN, CTYPE_OUT>(
                  [](CTYPE_IN v) { return static_cast<CTYPE_OUT>(v); },
                  [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },
                  out_ix);
              CTYPE_OUT mean = sum / static_cast<CTYPE_OUT>(num);
              mean_data[out_ix] = mean;
              // Pass 2: compute sum of squared deviations
              CTYPE_OUT sum2 = plan.execute<CTYPE_IN, CTYPE_OUT>(
                  [mean](CTYPE_IN v) {
                    return (
                        (static_cast<CTYPE_OUT>(v) - mean) *
                        (static_cast<CTYPE_OUT>(v) - mean));
                  },
                  [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },
                  out_ix);
              var_data[out_ix] = sum2 / denominator;
            }
          });
      ET_KERNEL_CHECK_MSG(ctx, success, Internal, , "parallel_for failed");
    } // !used_fast_path
  }
}

} // namespace

std::tuple<Tensor&, Tensor&> var_mean_correction_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    const optional<Scalar>& correction,
    bool keepdim,
    Tensor& out0,
    Tensor& out1) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&> ret_val(out0, out1);

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args(in, dim_list, keepdim, {}, out0),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args(in, dim_list, keepdim, {}, out1),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out0) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out1) == Error::Ok,
      InvalidArgument,
      ret_val);

  static constexpr auto name = "var_mean.correction_out";

  double correction_val = 1;
  if (correction.has_value()) {
    correction_val = utils::scalar_to<double>(correction.value());
  }

  const size_t num = get_reduced_dim_product(in, dim_list);
  const double denom = num - correction_val;

  ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, name, CTYPE_IN, [&] {
    ET_SWITCH_FLOATHBF16_TYPES(out0.scalar_type(), ctx, name, CTYPE_OUT, [&] {
      compute_var_mean<CTYPE_IN, CTYPE_OUT>(
          ctx, in, out0, out1, dim_list, num, denom);
    });
  });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
