/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

Tensor& sign_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(in, out);

  if (in.scalar_type() == exec_aten::ScalarType::Bool) {
    memcpy(out.mutable_data_ptr(), in.const_data_ptr(), in.nbytes());
  } else {
    ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "sign", CTYPE, [&] {
      apply_unary_map_fn(
          [](const CTYPE val_in) {
            if (std::isnan(val_in)) {
              return val_in;
            } else {
              return static_cast<CTYPE>((val_in > 0) - (val_in < 0));
            }
          },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
