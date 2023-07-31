/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

Tensor& logit_out(
    RuntimeContext& ctx,
    const Tensor& in,
    exec_aten::optional<double> eps,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  ET_CHECK_SAME_SHAPE2(in, out);

  ET_SWITCH_REAL_TYPES_AND(Bool, in.scalar_type(), ctx, "logit", CTYPE_IN, [&] {
    ET_SWITCH_FLOAT_TYPES(out.scalar_type(), ctx, "logit", CTYPE_OUT, [&] {
      apply_unary_map_fn(
          [eps](const CTYPE_IN val_in) {
            CTYPE_OUT xi = static_cast<CTYPE_OUT>(val_in);
            if (eps.has_value()) {
              if (xi < eps.value()) {
                xi = eps.value();
              } else if (xi > 1 - eps.value()) {
                xi = 1 - eps.value();
              }
            }
            ET_CHECK_MSG(xi > 0.0 && xi < 1.0, "input must be in (0, 1).");
            return static_cast<CTYPE_OUT>(
                log(xi / (static_cast<CTYPE_OUT>(1.0) - xi)));
          },
          in.const_data_ptr<CTYPE_IN>(),
          out.mutable_data_ptr<CTYPE_OUT>(),
          in.numel());
    });
  });
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
