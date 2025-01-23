/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& prod_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ScalarType> dtype,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_prod_out_args(in, dtype, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {}) == Error::Ok, InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  constexpr auto name = "prod.int_out";

  ET_SWITCH_REALHB_TYPES(in_type, ctx, name, CTYPE_IN, [&] {
    ET_SWITCH_REALHB_TYPES(out_type, ctx, name, CTYPE_OUT, [&] {
      const auto data_in = in.const_data_ptr<CTYPE_IN>();
      auto data_out = out.mutable_data_ptr<CTYPE_OUT>();
      data_out[0] = static_cast<CTYPE_OUT>(1);
      for (auto i = 0; i < in.numel(); ++i) {
        data_out[0] *= static_cast<CTYPE_OUT>(data_in[i]);
      }
    });
  });

  return out;
}

Tensor& prod_int_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args_single_dim(
          in, dim, keepdim, dtype, out, /*allow_empty_dim=*/true),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  constexpr auto name = "prod.int_out";

  ET_SWITCH_REALHB_TYPES(in_type, ctx, name, CTYPE_IN, [&] {
    ET_SWITCH_REALHB_TYPES(out_type, ctx, name, CTYPE_OUT, [&] {
      CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
      for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
        CTYPE_OUT prod = 1;
        if (in.numel() > 0) {
          std::tuple<CTYPE_OUT, long> acc =
              map_reduce_over_dim<CTYPE_IN, CTYPE_OUT>(
                  [](CTYPE_IN v) { return static_cast<CTYPE_OUT>(v); },
                  [](CTYPE_OUT outv, long, CTYPE_OUT acc, long) {
                    return std::tuple<CTYPE_OUT, long>{acc * outv, 0};
                  },
                  in,
                  dim,
                  out_ix);
          prod = std::get<0>(acc);
        }
        out_data[out_ix] = prod;
      }
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
