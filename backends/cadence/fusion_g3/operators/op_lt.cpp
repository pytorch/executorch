/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/fusion_g3/operators/xt_macros.h>
#include <executorch/kernels/portable/cpu/pattern/comparison_op.h>

using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;

namespace cadence {
namespace impl {
namespace G3 {
namespace native {

Tensor& lt_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
#ifdef OP_ARG_CHECK
  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(a, b, out),
      InvalidArgument,
      out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);
#endif

  int kTensorDimensionLimit = 5;

  int inp1_shape[kTensorDimensionLimit];
  int inp2_shape[kTensorDimensionLimit];
  int out_shape[kTensorDimensionLimit];

  bool broadcast = false;

  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  bool optimized = true;

  /* Added change to work with input dimensions more than 5 */
  for (int i = 0; i < max_dim; i++) {
    out_shape[i] = 1;
    inp1_shape[i] = 1;
    inp2_shape[i] = 1;
  }

  int offset_out = max_dim - out.dim();
  int offset_inp1 = max_dim - a.dim();
  int offset_inp2 = max_dim - b.dim();

  for (int i = 0; i < out.dim(); i++) {
    out_shape[i + offset_out] = out.size(i);
  }
  for (int i = 0; i < a.dim(); i++) {
    inp1_shape[i + offset_inp1] = a.size(i);
  }
  for (int i = 0; i < b.dim(); i++) {
    inp2_shape[i + offset_inp2] = b.size(i);
  }

  /*find broadcast*/
  for (int i = 0; i < max_dim; i++) {
    if (((inp1_shape[i]) != (out_shape[i])) ||
        ((inp2_shape[i]) != (out_shape[i]))) {
      broadcast = true;
    }
  }

  if (((broadcast) && (max_dim > kTensorDimensionLimit)) ||
      (!((a.scalar_type() == ScalarType::Float) &&
         (b.scalar_type() == ScalarType::Float) &&
         (out.scalar_type() == ScalarType::Bool)))) {
    optimized = false;
  }

  if (optimized) {
    const float* const inp1_data = a.const_data_ptr<float>();
    const float* const inp2_data = b.const_data_ptr<float>();
    signed char* const out_data = out.mutable_data_ptr<signed char>();

    if (b.numel() == 1) {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_less_scalar_f32xf32_bool,
          out_data,
          inp1_data,
          inp2_data[0],
          out.numel());
    } else if (broadcast) {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_less_broadcast_5D_f32xf32_bool,
          out_data,
          out_shape,
          inp1_data,
          inp1_shape,
          inp2_data,
          inp2_shape,
          max_dim);

    } else {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_less_f32xf32_bool,
          out_data,
          inp1_data,
          inp2_data,
          out.numel());
    }
  } else {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    static constexpr const char op_name[] = "lt.Tensor_out";
    torch::executor::native::internal::
        comparison_tensor_out<std::less, op_name>(ctx, a, b, out);
  }

  return out;
}

Tensor& lt_Scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
#ifdef OP_ARG_CHECK
  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(a, out),
      InvalidArgument,
      out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out);
#endif

  bool optimized = true;

  if (!((a.scalar_type() == ScalarType::Float) &&
        (out.scalar_type() == ScalarType::Bool))) {
    optimized = false;
  }

  if (optimized) {
    const float* const inp1_data = a.const_data_ptr<float>();
    float inp2_val;
    torch::executor::native::utils::extract_scalar(b, &inp2_val);

    signed char* const out_data = out.mutable_data_ptr<signed char>();

    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_elm_less_scalar_f32xf32_bool,
        out_data,
        inp1_data,
        inp2_val,
        out.numel());

  } else {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    static constexpr const char op_name[] = "lt.Scalar_out";
    torch::executor::native::internal::
        comparison_scalar_out<std::less, op_name>(ctx, a, b, out);
  }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
} // namespace cadence
