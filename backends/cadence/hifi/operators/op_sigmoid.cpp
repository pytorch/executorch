/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::aten::RuntimeContext;
using torch::executor::Error;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& sigmoid_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, in.scalar_type() != ScalarType::Bool, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensor_is_floating_type(out),
      InvalidArgument,
      out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();

  bool optimized = 1;
  if ((in_type != ScalarType::Float) || (out_type != ScalarType::Float))
    optimized = 0;

  if (optimized) {
    float* data_in = in.mutable_data_ptr<float>();
    float* data_out = out.mutable_data_ptr<float>();
    xa_nn_vec_sigmoid_f32_f32(data_out, data_in, in.numel());

    return out;
  }

  ET_SWITCH_REALHB_TYPES(in_type, ctx, "sigmoid.out", CTYPE_IN, [&]() {
    ET_SWITCH_FLOATH_TYPES(out_type, ctx, "sigmoid.out", CTYPE_OUT, [&]() {
      torch::executor::apply_unary_map_fn(
          [](const CTYPE_IN val_in) {
            // perform math in double to preserve precision
            double in_casted = static_cast<double>(val_in);
            double out_val = 1.0 / (1.0 + exp(-in_casted));
            return static_cast<CTYPE_OUT>(out_val);
          },
          in.const_data_ptr<CTYPE_IN>(),
          out.mutable_data_ptr<CTYPE_OUT>(),
          in.numel());
    });
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence