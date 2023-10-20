/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& any_all_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {}) == Error::Ok, InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, "any.all_out", CTYPE_IN, [&] {
    ET_SWITCH_TWO_TYPES(
        Bool, Byte, out_type, ctx, "any.all_out", CTYPE_OUT, [&] {
          const auto data_in = in.const_data_ptr<CTYPE_IN>();
          auto data_out = out.mutable_data_ptr<CTYPE_OUT>();
          data_out[0] = static_cast<CTYPE_OUT>(false);
          for (auto i = 0; i < in.numel(); ++i) {
            if (static_cast<CTYPE_OUT>(data_in[i])) {
              data_out[0] = static_cast<CTYPE_OUT>(true);
              break;
            }
          }
        });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
