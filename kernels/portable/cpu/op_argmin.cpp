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

using executorch::aten::Tensor;
using std::optional;

Tensor& argmin_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    std::optional<int64_t> dim,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_argmin_argmax_args(in, dim, keepdim, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "argmin.out";

  ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
    int64_t* out_data = out.mutable_data_ptr<int64_t>();

    const bool success = parallel_for_each_reduce_over_dim_output_index(
        in, dim, out, [&](const auto begin, const auto end) {
          for (const auto out_ix : c10::irange(begin, end)) {
            std::tuple<CTYPE, int64_t> acc = reduce_over_dim<CTYPE>(
                [](CTYPE v, int64_t ix, CTYPE acc_val, int64_t acc_ix) {
                  // the below condition as written is equivalent to
                  // !isnan(accval) && (isnan(v) || v < acc_val). cases:
                  // - if neither acc_val nor v is NaN, !(v >= acc_val) is
                  //   trivially equivalent to v < acc_val.
                  // - if acc_val is NaN, the whole thing is trivially false.
                  // - if acc_val is not NaN and v is NaN, then v >= acc_val
                  // - is false because all comparisons involving NaN are
                  // - false, so the result is true. The result is trivially
                  // - true for the above condition that uses isnan(v) as
                  // - well.
                  if (!utils::isnan_override(acc_val) && !(v >= acc_val)) {
                    acc_val = v;
                    acc_ix = ix;
                  }
                  return std::tuple<CTYPE, int64_t>{acc_val, acc_ix};
                },
                in,
                dim,
                out_ix);
            out_data[out_ix] = std::get<1>(acc);
          }
        });
    ET_KERNEL_CHECK_MSG(ctx, success, Internal, , "parallel_for failed");
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
