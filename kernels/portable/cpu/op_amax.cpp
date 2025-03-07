/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cmath>

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

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, "amax.out", CTYPE, [&]() {
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    for (const auto out_ix : c10::irange(out.numel())) {
      out_data[out_ix] = reduce_over_dim_list<CTYPE>(
          [](CTYPE v, CTYPE max_v) {
            return std::isnan(v) || v > max_v ? v : max_v;
          },
          in,
          dim_list,
          out_ix);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
