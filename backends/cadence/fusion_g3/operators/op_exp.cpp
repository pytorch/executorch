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
#include <xa_nnlib_kernels_api.h>

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::RuntimeContext;
using torch::executor::Error;

namespace cadence {
namespace impl {
namespace G3 { 
namespace native {

#define XT_KERNEL_CHECK(ctx, out, kernel, ...) \
  const auto ret = kernel(__VA_ARGS__);        \
  ET_KERNEL_CHECK_MSG(                         \
      ctx,                                     \
      ret == 0,                                \
      InvalidArgument,                         \
      out,                                     \
      "Failed to run kernel: " #kernel "(" #__VA_ARGS__ ")");

Tensor& exp_out(RuntimeContext& ctx,
                const Tensor& in,
                Tensor& out)
{
#ifdef OP_ARG_CHECK
    ET_KERNEL_CHECK(ctx, executorch::runtime::tensor_is_floating_type(out), InvalidArgument, out);

    // Resize for dynamic shape
    ET_KERNEL_CHECK_MSG(
      ctx,
      executorch::runtime::resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

    ET_KERNEL_CHECK(
      ctx, executorch::runtime::tensors_have_same_dim_order(in, out),
      InvalidArgument, out);
#endif

    if(out.scalar_type() == ScalarType::Float)
    {
        float * const out_data = out.mutable_data_ptr<float>();
        const float * const in_data = in.const_data_ptr<float>();

        XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_exp_f32_f32,
          out_data,
          in_data,
          out.numel());

        return out;
    }
    else
    {
        return torch::executor::native::internal::unary_ufunc_realhbbf16_to_floathbf16(std::exp, ctx, in, out);
    }
}

} // namespace native
} // namespace G3
} // namespace impl
} // namespace cadence