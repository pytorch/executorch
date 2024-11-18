/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

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

  ET_SWITCH_REALHB_TYPES(in.scalar_type(), ctx, "mean.out", CTYPE_IN, [&] {
    ET_SWITCH_FLOATH_TYPES(out.scalar_type(), ctx, "mean.out", CTYPE_OUT, [&] {
      CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
      const size_t num = get_reduced_dim_product(in, dim_list);
      for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
        CTYPE_OUT sum = 0;
        if (in.numel() > 0) {
          sum = map_reduce_over_dim_list<CTYPE_IN, CTYPE_OUT>(
              [](CTYPE_IN v) { return static_cast<CTYPE_OUT>(v); },
              [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },
              in,
              dim_list,
              out_ix);
        }
        out_data[out_ix] = sum / static_cast<float>(num);
      }
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
