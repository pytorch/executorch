/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <xa_nnlib_kernels_api.h>

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::canCast;
using torch::executor::Error;
using torch::executor::KernelRuntimeContext;

namespace cadence {
namespace impl {
namespace G3 {
namespace native {

Tensor& add_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type =
      executorch::runtime::promoteTypes(a.scalar_type(), b.scalar_type());

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (canCast(common_type, out.scalar_type()) &&
       torch::executor::check_alpha_type(
           torch::executor::native::utils::get_scalar_dtype(alpha),
           common_type)),
      InvalidArgument,
      out);

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

  // Compute Dtype
  ScalarType compute_type =
      torch::executor::native::utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "add.out";

  const exec_aten::ArrayRef<Tensor::SizesType> a_size = a.sizes();
  const exec_aten::ArrayRef<Tensor::SizesType> b_size = b.sizes();
  const exec_aten::ArrayRef<Tensor::SizesType> out_size = out.sizes();

  int kTensorDimensionLimit = 5;

  int inp1_shape[kTensorDimensionLimit];
  int inp2_shape[kTensorDimensionLimit];
  int out_shape[kTensorDimensionLimit];

  /* input shapes and output shapes */
  for (auto i = 0; i < a_size.size(); i++) {
    inp1_shape[i] = a_size[i];
  }

  for (auto i = 0; i < b_size.size(); i++) {
    inp2_shape[i] = b_size[i];
  }

  for (auto i = 0; i < out_size.size(); i++) {
    out_shape[i] = out_size[i];
  }

  /*find broadcast*/
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted || b_is_broadcasted);

  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();

  if (compute_type == ScalarType::Int) {
    const int* const inp1_data = a.const_data_ptr<int>();
    const int* const inp2_data = b.const_data_ptr<int>();
    int* const out_data = out.mutable_data_ptr<int>();

    int alpha_val;
    torch::executor::native::utils::extract_scalar(alpha, &alpha_val);
    if (broadcast) {
      xa_nn_elm_add_broadcast_5D_32x32_32(
          out_data,
          out_shape,
          inp1_data,
          inp1_shape,
          inp2_data,
          inp2_shape,
          max_dim,
          alpha_val);
    } else {
      xa_nn_elm_add_32x32_32(
          out_data, inp1_data, inp2_data, alpha_val, out.numel());
    }
  } else if (compute_type == ScalarType::Float) {
    const float* const inp1_data = a.const_data_ptr<float>();
    const float* const inp2_data = b.const_data_ptr<float>();
    float* const out_data = out.mutable_data_ptr<float>();

    float alpha_val;
    torch::executor::native::utils::extract_scalar(alpha, &alpha_val);

    if (broadcast) {
      xa_nn_elm_add_broadcast_5D_f32xf32_f32(
          out_data,
          out_shape,
          inp1_data,
          inp1_shape,
          inp2_data,
          inp2_shape,
          max_dim,
          alpha_val);
    } else {
      xa_nn_elm_add_f32xf32_f32(
          out_data, inp1_data, inp2_data, alpha_val, out.numel());
    }
  } else {
    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      const CTYPE_COMPUTE val_alpha =
          torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(alpha);
      torch::executor::native::utils::apply_bitensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name>(
          [val_alpha](const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
            return val_a + val_alpha * val_b;
          },
          ctx,
          a,
          torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
          b,
          torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
          out,
          torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16);
    });
  }

  return out;
}

Tensor& add_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type =
      torch::executor::native::utils::promote_type_with_scalar(
          a.scalar_type(), b);

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (common_type == out.scalar_type() &&
       torch::executor::check_alpha_type(
           torch::executor::native::utils::get_scalar_dtype(alpha),
           common_type)),
      InvalidArgument,
      out);

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

  // Compute Dtype
  ScalarType compute_type =
      torch::executor::native::utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "add.Scalar_out";

  if (compute_type == ScalarType::Int) {
    const int* const inp1_data = a.const_data_ptr<int>();
    int inp2_val;
    torch::executor::native::utils::extract_scalar(b, &inp2_val);

    int alpha_val;
    torch::executor::native::utils::extract_scalar(alpha, &alpha_val);

    int* const out_data = out.mutable_data_ptr<int>();

    xa_nn_elm_add_scalar_32x32_32(
        out_data, inp1_data, inp2_val, alpha_val, out.numel());
  } else if (compute_type == ScalarType::Float) {
    const float* const inp1_data = a.const_data_ptr<float>();
    float inp2_val;
    torch::executor::native::utils::extract_scalar(b, &inp2_val);

    float alpha_val;
    torch::executor::native::utils::extract_scalar(alpha, &alpha_val);

    float* const out_data = out.mutable_data_ptr<float>();

    xa_nn_elm_add_scalar_f32xf32_f32(
        out_data, inp1_data, inp2_val, alpha_val, out.numel());
  } else {
    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      torch::executor::native::utils::
          apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
              [b, alpha](const CTYPE_COMPUTE val_a) {
                CTYPE_COMPUTE val_b =
                    torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(b);
                CTYPE_COMPUTE val_alpha =
                    torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(
                        alpha);
                return val_a + val_alpha * val_b;
              },
              ctx,
              a,
              torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
              out,
              torch::executor::native::utils::SupportedTensorDtypes::
                  SAME_AS_COMMON);
    });
  }
  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
} // namespace cadence