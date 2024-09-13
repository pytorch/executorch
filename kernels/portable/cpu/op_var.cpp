/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {
namespace {

template <typename CTYPE_IN, typename CTYPE_OUT>
void compute_variance(
    const Tensor& in,
    Tensor& out,
    optional<ArrayRef<int64_t>> dim_list,
    const size_t num,
    const double denominator) {
  CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
  if (num == 0 || denominator <= 0) {
    for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
      out_data[out_ix] = NAN;
    }
  } else {
    for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
      CTYPE_OUT sum = map_reduce_over_dim_list<CTYPE_IN, CTYPE_OUT>(
          [](CTYPE_IN v) { return static_cast<CTYPE_OUT>(v); },
          [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },
          in,
          dim_list,
          out_ix);
      CTYPE_OUT mean = sum / num;
      CTYPE_OUT sum2 = map_reduce_over_dim_list<CTYPE_IN, CTYPE_OUT>(
          [mean](CTYPE_IN v) {
            return (
                (static_cast<CTYPE_OUT>(v) - mean) *
                (static_cast<CTYPE_OUT>(v) - mean));
          },
          [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },
          in,
          dim_list,
          out_ix);
      out_data[out_ix] = sum2 / denominator;
    }
  }
}

} // namespace

Tensor& var_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool unbiased,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args(in, dim_list, keepdim, {}, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(in), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  const size_t num = get_reduced_dim_product(in, dim_list);
  const size_t denom = unbiased ? num - 1 : num;

  constexpr auto name = "var.out";

  ET_SWITCH_FLOAT_TYPES(in.scalar_type(), ctx, name, CTYPE_IN, [&] {
    ET_SWITCH_FLOAT_TYPES(out.scalar_type(), ctx, name, CTYPE_OUT, [&] {
      compute_variance<CTYPE_IN, CTYPE_OUT>(in, out, dim_list, num, denom);
    });
  });

  return out;
}

Tensor& var_correction_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    const optional<Scalar>& correction,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args(in, dim_list, keepdim, {}, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "var.correction_out";

  double correction_val = 1;
  if (correction.has_value()) {
    ScalarType corr_type = utils::get_scalar_dtype(correction.value());
    ET_SWITCH_SCALAR_OBJ_TYPES(corr_type, ctx, name, CTYPE_CORR, [&]() {
      CTYPE_CORR corr_val = 0;
      utils::extract_scalar(correction.value(), &corr_val);
      correction_val = static_cast<double>(corr_val);
    });
  }

  const size_t num = get_reduced_dim_product(in, dim_list);
  const double denom = num - correction_val;

  ET_SWITCH_FLOAT_TYPES(in.scalar_type(), ctx, name, CTYPE_IN, [&] {
    ET_SWITCH_FLOAT_TYPES(out.scalar_type(), ctx, name, CTYPE_OUT, [&] {
      compute_variance<CTYPE_IN, CTYPE_OUT>(in, out, dim_list, num, denom);
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
