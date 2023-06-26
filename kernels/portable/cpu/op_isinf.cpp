// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cmath>

#include <executorch/core/Assert.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

Tensor& isinf_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  ET_CHECK_SAME_SHAPE2(in, out);
  ET_CHECK_MSG(
      out.scalar_type() == exec_aten::ScalarType::Bool,
      "Expected out tensor to have dtype Bool, but got %hhd instead.",
      out.scalar_type());

  ET_SWITCH_REAL_TYPES_AND(Bool, in.scalar_type(), ctx, "isinf", CTYPE_IN, [&] {
    apply_unary_map_fn(
        [](const CTYPE_IN val_in) { return std::isinf(val_in); },
        in.const_data_ptr<CTYPE_IN>(),
        out.mutable_data_ptr<bool>(),
        in.numel());
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
