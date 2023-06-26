// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cmath>

#include <executorch/core/Assert.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

Tensor& acos_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  ET_CHECK_SAME_SHAPE2(in, out);

  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "acos", CTYPE_IN, [&] {
    ET_SWITCH_FLOAT_TYPES(out.scalar_type(), ctx, "acos", CTYPE_OUT, [&] {
      apply_unary_map_fn(
          [](const CTYPE_IN val_in) {
            CTYPE_OUT xi = static_cast<CTYPE_OUT>(val_in);
            return static_cast<CTYPE_OUT>(acos(xi));
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
