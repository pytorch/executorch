/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& any_all_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_CHECK_MSG(out.dim() == 0, "dimension of the output Tensor shall be 0.");

  ET_SWITCH_REAL_TYPES_AND(Bool, in.scalar_type(), ctx, "any", CTYPE_IN, [&] {
    ET_SWITCH_TWO_TYPES(
        Bool, Byte, out.scalar_type(), ctx, "any", CTYPE_OUT, [&] {
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
