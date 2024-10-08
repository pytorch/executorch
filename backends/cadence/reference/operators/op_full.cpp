/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& full_out(
    KernelRuntimeContext& ctx,
    const IntArrayRef sizes,
    const Scalar& fill_value,
    Tensor& out) {
  (void)ctx;

  ScalarType val_type = utils::get_scalar_dtype(fill_value);
  ScalarType out_type = out.scalar_type();

  Error err = resize_tensor(out, sizes);
  ET_CHECK_MSG(err == Error::Ok, "Could not resize out");

  ET_SWITCH_REAL_TYPES_AND(Bool, val_type, ctx, "full", CTYPE_VAL, [&] {
    CTYPE_VAL val;
    ET_EXTRACT_SCALAR(fill_value, val);

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
} // namespace executor
} // namespace torch
