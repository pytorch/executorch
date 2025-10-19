/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>

#include <cmath>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/fusion_g3/operators/xt_macros.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;

namespace impl {
namespace G3 {
namespace native {

Tensor& sigmoid_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

#ifdef OP_ARG_CHECK
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensor_is_floating_type(out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(in, out),
      InvalidArgument,
      out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      executorch::runtime::resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");
#endif

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "sigmoid.out";

  if ((in.scalar_type() == ScalarType::Float) &&
      (out.scalar_type() == ScalarType::Float)) {
    const float* const in_data = in.const_data_ptr<float>();
    float* const out_data = out.mutable_data_ptr<float>();

    XT_KERNEL_CHECK(
        ctx, out, xa_nn_sigmoid_f32_f32, out_data, in_data, out.numel());
  } else {
    ET_KERNEL_CHECK(
        ctx, in.scalar_type() != ScalarType::Bool, InvalidArgument, out);

    ScalarType compute_type =
        executorch::runtime::isFloatingType(in.scalar_type())
        ? in.scalar_type()
        : ScalarType::Float;
    compute_type =
        torch::executor::native::utils::get_compute_type(compute_type);
    ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      torch::executor::native::utils::apply_unitensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name>(
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
  }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
