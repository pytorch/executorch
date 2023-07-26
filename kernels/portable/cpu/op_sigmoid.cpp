// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>

#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& sigmoid_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  Error err = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(err == Error::Ok, "Could not resize output");

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, "sigmoid", CTYPE_IN, [&]() {
    ET_SWITCH_FLOAT_TYPES(out_type, ctx, "sigmoid", CTYPE_OUT, [&]() {
      apply_unary_map_fn(
          [](const CTYPE_IN val_in) {
            // perform math in double to preserve precision
            double in_casted = static_cast<double>(val_in);
            double out_val = 1.0 / (1.0 + exp(-in_casted));
            return static_cast<CTYPE_OUT>(out_val);
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
