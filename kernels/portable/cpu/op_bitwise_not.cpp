// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cmath>

#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

/**
 * Computes the bitwise NOT of the given input tensor. The input tensor must be
 * of Integral or Boolean types. For bool tensors, it computes the logical NOT.
 **/
Tensor& bitwise_not_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(in, out);

  if (in.scalar_type() == exec_aten::ScalarType::Bool) {
    apply_unary_map_fn(
        [](const bool val_in) { return !val_in; },
        in.const_data_ptr<bool>(),
        out.mutable_data_ptr<bool>(),
        in.numel());
  } else if (isIntegralType(in.scalar_type(), /*includeBool=*/false)) {
    ET_SWITCH_INT_TYPES(in.scalar_type(), ctx, "bitwise_not", CTYPE, [&] {
      apply_unary_map_fn(
          [](const CTYPE val_in) { return ~val_in; },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    });
  } else {
    ET_CHECK_MSG(false, "Unsupported input dtype %hhd", in.scalar_type());
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
