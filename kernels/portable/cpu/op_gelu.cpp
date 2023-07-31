/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/math_constants.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using string_view = exec_aten::string_view;

Tensor& gelu_out(
    RuntimeContext& ctx,
    const Tensor& in,
    string_view approximate,
    Tensor& out) {
  (void)ctx;

  Error err = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(err == Error::Ok, "Could not resize output");

  ET_CHECK_SAME_SHAPE_AND_DTYPE2(in, out);

  ET_SWITCH_FLOAT_TYPES(in.scalar_type(), ctx, "gelu", CTYPE, [&]() {
    if (approximate == "tanh") {
      apply_unary_map_fn(
          [](const CTYPE x) {
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
          [](const CTYPE x) { return 0.5 * x * (1 + std::erf(x * M_SQRT1_2)); },
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
