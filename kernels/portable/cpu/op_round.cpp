// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>

#include <executorch/core/Assert.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

namespace {

// Rounds a floating point value to the closest integer. Values with a
// fractional part of exactly 0.5 are rounded to the closest even integer. Uses
// the implementation from torch/src/jit/runtime/register_ops_utils.h.
template <typename CTYPE>
inline CTYPE round_to_even(CTYPE a) {
  return a - std::floor(a) == 0.5 ? (std::round(a * 0.5) * 2.0) : std::round(a);
}

} // namespace

Tensor& round_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(in, out);

  auto in_scalar_type = in.scalar_type();

  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "round", CTYPE, [&] {
    apply_unary_map_fn(
        [in_scalar_type](const CTYPE val_in) {
          if (isIntegralType(in_scalar_type, /*includeBool=*/false)) {
            return val_in;
          } else {
            return static_cast<CTYPE>(round_to_even<CTYPE>(val_in));
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
