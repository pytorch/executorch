/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cmath>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {
namespace {

template <typename CTYPE>
constexpr CTYPE upper_bound() {
  using lim = std::numeric_limits<CTYPE>;
  return lim::has_infinity ? lim::infinity() : lim::max();
}

} // namespace

using ScalarType = executorch::aten::ScalarType;
using SizesType = executorch::aten::SizesType;
using Tensor = executorch::aten::Tensor;

std::tuple<Tensor&, Tensor&> min_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    Tensor& min,
    Tensor& min_indices) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_min_max_args(in, dim, keepdim, min, min_indices),
      InvalidArgument,
      (std::tuple<Tensor&, Tensor&>({min, min_indices})));

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim, keepdim, min) == Error::Ok,
      InvalidArgument,
      (std::tuple<Tensor&, Tensor&>({min, min_indices})));

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(min_indices, min.sizes()) == Error::Ok,
      InvalidArgument,
      (std::tuple<Tensor&, Tensor&>({min, min_indices})));

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, min),
      InvalidArgument,
      (std::tuple<Tensor&, Tensor&>({min, min_indices})));

  ET_KERNEL_CHECK(
      ctx,
      tensor_is_default_dim_order(min_indices),
      InvalidArgument,
      (std::tuple<Tensor&, Tensor&>({min, min_indices})));

  ET_KERNEL_CHECK(
      ctx,
      tensor_is_default_dim_order(in),
      InvalidArgument,
      (std::tuple<Tensor&, Tensor&>({min, min_indices})));

  dim = dim < 0 ? dim + in.dim() : dim;

  ET_SWITCH_REALHBBF16_TYPES(
      in.scalar_type(), ctx, "min.dim_min", CTYPE, [&]() {
        CTYPE* min_data = min.mutable_data_ptr<CTYPE>();
        int64_t* min_indices_data = min_indices.mutable_data_ptr<int64_t>();

        const bool success = parallel_for_each_reduce_over_dim_output_index(
            in, dim, min, [&](const auto begin, const auto end) {
              for (const auto out_ix : c10::irange(begin, end)) {
                std::tuple<CTYPE, int64_t> acc = reduce_over_dim<CTYPE>(
                    [](CTYPE v, int64_t ix, CTYPE acc_val, int64_t acc_ix) {
                      if (!utils::isnan_override(acc_val) &&
                          (utils::isnan_override(v) || v < acc_val)) {
                        acc_val = v;
                        acc_ix = ix;
                      }
                      return std::tuple<CTYPE, int64_t>{acc_val, acc_ix};
                    },
                    in,
                    dim,
                    out_ix);
                min_data[out_ix] = std::get<0>(acc);
                min_indices_data[out_ix] = std::get<1>(acc);
              }
            });
        ET_KERNEL_CHECK_MSG(ctx, success, Internal, , "parallel_for failed");
      });

  return {min, min_indices};
}

Tensor&
min_unary_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {}) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(in_type, out_type), InvalidArgument, out);

  static constexpr auto name = "min.unary_out";

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, name, CTYPE_IN, [&] {
    ET_SWITCH_REALHBBF16_TYPES(out_type, ctx, name, CTYPE_OUT, [&] {
      const auto data_in = in.const_data_ptr<CTYPE_IN>();
      auto data_out = out.mutable_data_ptr<CTYPE_OUT>();
      data_out[0] = upper_bound<CTYPE_OUT>();
      for (const auto i : c10::irange(in.numel())) {
        CTYPE_OUT val = static_cast<CTYPE_OUT>(data_in[i]);
        if (utils::isnan_override(val)) {
          data_out[0] = val;
          break;
        }
        if (val < data_out[0]) {
          data_out[0] = val;
        }
      }
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
