/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::IntArrayRef;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::Error;
using torch::executor::native::utils::extract_scalar;
using torch::executor::native::utils::get_scalar_dtype;

namespace impl {
namespace vision {
namespace native {

Tensor& full_out(
    KernelRuntimeContext& ctx,
    const IntArrayRef sizes,
    const Scalar& fill_value,
    Tensor& out) {
  (void)ctx;

  ScalarType val_type = get_scalar_dtype(fill_value);
  ScalarType out_type = out.scalar_type();

  Error err = resize_tensor(out, sizes);
  ET_CHECK_MSG(err == Error::Ok, "Could not resize out");

  ET_SWITCH_REAL_TYPES_AND(Bool, val_type, ctx, "full", CTYPE_VAL, [&] {
    CTYPE_VAL val;
    ET_CHECK_MSG(
        extract_scalar(fill_value, &val),
        "Could not be extracted: wrong type or out of range");

    ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "full", CTYPE_OUT, [&] {
      CTYPE_OUT val_casted = static_cast<CTYPE_OUT>(val);
      auto data_out = out.mutable_data_ptr<CTYPE_OUT>();
      for (size_t i = 0; i < out.numel(); ++i) {
        data_out[i] = val_casted;
      }
    });
  });

  return out;
}

} // namespace native
} // namespace vision
} // namespace impl
