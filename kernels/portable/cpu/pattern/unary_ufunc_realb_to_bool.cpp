// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/core/function_ref.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

Tensor& unary_ufunc_realb_to_bool(
    FunctionRef<bool(double)> fn,
    RuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  ET_CHECK_MSG(
      out.scalar_type() == exec_aten::ScalarType::Bool,
      "Expected out tensor to have dtype Bool, but got %hhd instead.",
      out.scalar_type());

  const auto in_type = in.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, __func__, CTYPE_IN, [&] {
    apply_unary_map_fn(
        [fn](const CTYPE_IN val_in) { return fn(val_in); },
        in.const_data_ptr<CTYPE_IN>(),
        out.mutable_data_ptr<bool>(),
        in.numel());
  });

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
