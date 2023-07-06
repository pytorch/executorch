// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>

#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& hardtanh_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Scalar& min,
    const Scalar& max,
    Tensor& out) {
  (void)ctx;

  Error err = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(err == Error::Ok, "Could not resize output");

  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "hardtanh", CTYPE, [&]() {
    CTYPE min_val = 0;
    ET_EXTRACT_SCALAR(min, min_val);

    CTYPE max_val = 0;
    ET_EXTRACT_SCALAR(max, max_val);

    apply_unary_map_fn(
        [min_val, max_val](const CTYPE val_in) {
          if (val_in > max_val) {
            return max_val;
          } else if (val_in < min_val) {
            return min_val;
          } else {
            return val_in;
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
