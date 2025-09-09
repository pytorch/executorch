/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

Tensor& sigmoid_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType compute_type =
      executorch::runtime::isFloatingType(in.scalar_type()) ? in.scalar_type()
                                                            : ScalarType::Float;
  compute_type = utils::get_compute_type(compute_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "sigmoid.out";

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_unitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::FLOATHBF16>(
        [](const auto val_in) {
          const auto one = static_cast<decltype(val_in)>(1.0);
          auto out_val = one / (one + executorch::math::exp(-val_in));
          return out_val;
        },
        ctx,
        in,
        utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
