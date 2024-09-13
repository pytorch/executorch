/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::optional;
using exec_aten::Tensor;

Tensor& argmax_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<int64_t> dim,
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

  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "argmax.out", CTYPE, [&] {
    long* out_data = out.mutable_data_ptr<long>();

    for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
      std::tuple<CTYPE, long> acc = reduce_over_dim<CTYPE>(
          [](CTYPE v, long ix, CTYPE acc_val, long acc_ix) {
            if (!std::isnan(acc_val) && (std::isnan(v) || v > acc_val)) {
              acc_val = v;
              acc_ix = ix;
            }
            return std::tuple<CTYPE, long>{acc_val, acc_ix};
          },
          in,
          dim,
          out_ix);
      out_data[out_ix] = std::get<1>(acc);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
