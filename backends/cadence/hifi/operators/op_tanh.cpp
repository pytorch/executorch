/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>
#include "kernels.h"

namespace torch {
namespace executor {
namespace native {

Tensor& tanh_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {

  int fall_back = 0;
  if((in.scalar_type() != ScalarType::Float) || (out.scalar_type() != ScalarType::Float))
      fall_back = 1;
  
  if(!fall_back)
  {
    float* data_in = in.mutable_data_ptr<float>();
    float* data_out = out.mutable_data_ptr<float>();
    xa_nn_vec_tanh_f32_f32(data_out, data_in, (int)in.numel());
    return out;
  }
  else
  {
    return internal::unary_ufunc_realhb_to_floath(std::tanh, ctx, in, out);
  }

}

} // namespace native
} // namespace executor
} // namespace torch
