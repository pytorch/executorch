// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& leaky_relu_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Scalar& negative_slope,
    Tensor& out) {
  (void)ctx;

  Error err = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(err == Error::Ok, "Could not resize output");

  ScalarType in_type = in.scalar_type();
  ScalarType sc_type = utils::get_scalar_dtype(negative_slope);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(in_type == out_type);

  ET_SWITCH_FLOAT_TYPES(in_type, ctx, "leaky_relu", CTYPE, [&]() {
    CTYPE negative_slope_casted;
    ET_SWITCH_SCALAR_OBJ_TYPES(sc_type, ctx, "leaky_relu", CTYPE_MIN, [&]() {
      CTYPE_MIN negative_slope_val;
      ET_EXTRACT_SCALAR(negative_slope, negative_slope_val);
      negative_slope_casted = static_cast<CTYPE>(negative_slope_val);
    });

    apply_unary_map_fn(
        [negative_slope_casted](const CTYPE val_in) {
          if (val_in >= 0) {
            return val_in;
          } else {
            return val_in * negative_slope_casted;
          }
        },
        in.const_data_ptr<CTYPE>(),
        out.mutable_data_ptr<CTYPE>(),
        in.numel());
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
