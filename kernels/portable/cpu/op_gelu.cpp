/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/math_constants.h>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using string_view = exec_aten::string_view;

Tensor& gelu_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    string_view approximate,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_gelu_args(in, approximate, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_SWITCH_FLOAT_TYPES(in.scalar_type(), ctx, "gelu.out", CTYPE, [&]() {
    if (approximate == "tanh") {
      apply_unary_map_fn(
          [](const CTYPE x) {
            if (x == -std::numeric_limits<CTYPE>::infinity()) {
              return static_cast<CTYPE>(0.0);
            } else if (x == std::numeric_limits<CTYPE>::infinity()) {
              return std::numeric_limits<CTYPE>::infinity();
            }
            const CTYPE kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
            const CTYPE kKappa = static_cast<float>(0.044715);

            const CTYPE x_cubed = x * x * x;
            const CTYPE inner = kBeta * (x + kKappa * x_cubed);
            const CTYPE ret = 0.5 * x * (1 + std::tanh(inner));

            return ret;
          },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    } else if (approximate == "none") {
      apply_unary_map_fn(
          [](const CTYPE x) {
            if (x == -std::numeric_limits<CTYPE>::infinity()) {
              return static_cast<CTYPE>(0.0);
            } else if (x == std::numeric_limits<CTYPE>::infinity()) {
              return std::numeric_limits<CTYPE>::infinity();
            }
            return static_cast<CTYPE>(0.5 * x * (1 + std::erf(x * M_SQRT1_2)));
          },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    } else {
      ET_CHECK_MSG(
          false,
          "Invalid approximation format: %.*s for gelu",
          static_cast<int>(approximate.length()),
          approximate.data());
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
