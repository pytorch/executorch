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
#include <executorch/backends/cadence/hifi/kernels/kernels.h>

namespace torch {
namespace executor {
namespace native {

Tensor& tanh_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {

  bool optimized = 1;
  if((in.scalar_type() != ScalarType::Float) || (out.scalar_type() != ScalarType::Float))
      optimized = 0;
  
  if(optimized)
  {
    float* data_in = in.mutable_data_ptr<float>();
    float* data_out = out.mutable_data_ptr<float>();
    xa_nn_vec_tanh_f32_f32(data_out, data_in, (int)in.numel());
    return out;
  }

  return internal::unary_ufunc_realhb_to_floath(std::tanh, ctx, in, out);

}

} // namespace native
} // namespace executor
} // namespace torch
