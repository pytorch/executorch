// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/optimized/vec/vec.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& opt_neg_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "neg", CTYPE, [&] {
    using Vec = executorch::vec::Vectorized<CTYPE>;
    executorch::vec::map<CTYPE>(
        [](Vec x) { return x.neg(); },
        out.mutable_data_ptr<CTYPE>(),
        in.const_data_ptr<CTYPE>(),
        in.numel());
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
