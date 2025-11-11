/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::RuntimeContext;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::Error;

namespace impl {
namespace HiFi {
namespace native {

using Tensor = executorch::aten::Tensor;

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

  ScalarType compute_type =
      executorch::runtime::isFloatingType(in.scalar_type()) ? in.scalar_type()
                                                            : ScalarType::Float;
  compute_type = torch::executor::native::utils::get_compute_type(compute_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "sigmoid.out";

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    torch::executor::native::utils::
        apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
            [](const CTYPE_COMPUTE val_in) {
              CTYPE_COMPUTE out_val = static_cast<CTYPE_COMPUTE>(1.0) /
                  (static_cast<CTYPE_COMPUTE>(1.0) + exp(-val_in));
              return out_val;
            },
            ctx,
            in,
            torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
            out,
            torch::executor::native::utils::SupportedTensorDtypes::FLOATHBF16);
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
