/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using ScalarType = exec_aten::ScalarType;
using SizesType = exec_aten::SizesType;
using Tensor = exec_aten::Tensor;

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

  ET_SWITCH_REAL_TYPES_AND(
      Bool, in.scalar_type(), ctx, "min.dim_min", CTYPE, [&]() {
        CTYPE* min_data = min.mutable_data_ptr<CTYPE>();
        long* min_indices_data = min_indices.mutable_data_ptr<long>();

        for (size_t out_ix = 0; out_ix < min.numel(); ++out_ix) {
          std::tuple<CTYPE, long> acc = reduce_over_dim<CTYPE>(
              [](CTYPE v, long ix, CTYPE acc_val, long acc_ix) {
                if (!std::isnan(acc_val) && (std::isnan(v) || v < acc_val)) {
                  acc_val = v;
                  acc_ix = ix;
                }
                return std::tuple<CTYPE, long>{acc_val, acc_ix};
              },
              in,
              dim,
              out_ix);
          min_data[out_ix] = std::get<0>(acc);
          min_indices_data[out_ix] = std::get<1>(acc);
        }
      });

  return {min, min_indices};
}

} // namespace native
} // namespace executor
} // namespace torch
