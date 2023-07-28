/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <cstdint>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

Tensor&
scalar_tensor_out(RuntimeContext& context, const Scalar& s, Tensor& out) {
  (void)context;

  ET_CHECK_MSG(out.numel() == 1, "Output tensor must have only one element");

  ScalarType s_type = utils::get_scalar_dtype(s);
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "scalar_tensor", CTYPE, [&]() {
    ET_SWITCH_SCALAR_OBJ_TYPES(s_type, ctx, "scalar_tensor", CTYPE_S, [&]() {
      CTYPE_S val_s;
      ET_EXTRACT_SCALAR(s, val_s);
      out.mutable_data_ptr<CTYPE>()[0] = val_s;
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
