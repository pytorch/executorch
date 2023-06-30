// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/core/FunctionRef.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

Tensor& unary_ufunc_realb_to_float(
    FunctionRef<double(double)> fn,
    RuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  const auto in_type = in.scalar_type();
  const auto out_type = out.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, __func__, CTYPE_IN, [&] {
    ET_SWITCH_FLOAT_TYPES(out_type, ctx, __func__, CTYPE_OUT, [&] {
      apply_unary_map_fn(
          [fn](const CTYPE_IN val_in) {
            CTYPE_OUT xi = static_cast<CTYPE_OUT>(val_in);
            return static_cast<CTYPE_OUT>(fn(xi));
          },
          in.const_data_ptr<CTYPE_IN>(),
          out.mutable_data_ptr<CTYPE_OUT>(),
          in.numel());
    });
  });

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
