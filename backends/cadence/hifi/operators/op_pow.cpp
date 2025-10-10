/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::can_cast;
using executorch::runtime::canCast;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::promoteTypes;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;
using torch::executor::native::utils::apply_bitensor_elementwise_fn;
using torch::executor::native::utils::apply_unitensor_elementwise_fn;
using torch::executor::native::utils::get_compute_type;
using torch::executor::native::utils::promote_type_with_scalar;
using torch::executor::native::utils::scalar_to;
using torch::executor::native::utils::SupportedTensorDtypes;

namespace impl {
namespace HiFi {
namespace native {

Tensor& pow_Tensor_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promoteTypes(a.scalar_type(), b.scalar_type());

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (canCast(common_type, out.scalar_type()) &&
       common_type != ScalarType::Bool),
      InvalidArgument,
      out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  // Compute Dtype
  ScalarType compute_type = get_compute_type(common_type);
  if (compute_type != ScalarType::Float) {
    compute_type = ScalarType::Double;
  }

  constexpr int kNnlibMaxDim = 16;
  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  bool optimized = true;

  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted && b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  ScalarType out_type = out.scalar_type();

  if (out_type != ScalarType::Float)
    optimized = false;

  if (max_dim > kNnlibMaxDim)
    optimized = false;

  WORD32 num_elm = out.numel();

  if (optimized) {
    if (broadcast) {
      WORD32* __restrict__ ptr1 =
          (WORD32* __restrict__)kernels::allocate_temp_memory(
              ctx, num_elm * sizeof(int));

      ET_KERNEL_CHECK(ctx, ptr1 != nullptr, MemoryAllocationFailed, out);

      WORD32* __restrict__ ptr2 =
          (WORD32* __restrict__)kernels::allocate_temp_memory(
              ctx, num_elm * sizeof(int));

      ET_KERNEL_CHECK(ctx, ptr2 != nullptr, MemoryAllocationFailed, out);

      WORD32* __restrict__ pin1 =
          (WORD32* __restrict__)a.const_data_ptr<float>();
      WORD32* __restrict__ pin2 =
          (WORD32* __restrict__)b.const_data_ptr<float>();

      WORD32 p_out_shape[kNnlibMaxDim];
      WORD32 p_inp1_shape[kNnlibMaxDim];
      WORD32 p_inp2_shape[kNnlibMaxDim];

      for (int i = 0; i < out_dim; i++)
        p_out_shape[i] = out.size(i);
      for (int i = 0; i < a_dim; i++)
        p_inp1_shape[i] = a.size(i);
      for (int i = 0; i < b_dim; i++)
        p_inp2_shape[i] = b.size(i);

      xa_nn_broadcast_32_32(ptr1, p_out_shape, pin1, p_inp1_shape, out_dim);

      xa_nn_broadcast_32_32(ptr2, p_out_shape, pin2, p_inp2_shape, out_dim);

      FLOAT32* __restrict__ p_out =
          (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp1 = (const FLOAT32* __restrict__)ptr1;
      const FLOAT32* __restrict__ p_inp2 = (const FLOAT32* __restrict__)ptr2;

      xa_nn_elm_pow_f32(p_out, p_inp1, p_inp2, num_elm);

    } else if (a_is_broadcasted && (!b_is_broadcasted)) {
      FLOAT32* __restrict__ ptr1 =
          (FLOAT32* __restrict__)kernels::allocate_temp_memory(
              ctx, num_elm * sizeof(int));

      ET_KERNEL_CHECK(ctx, ptr1 != nullptr, MemoryAllocationFailed, out);

      FLOAT32* __restrict__ pin1 =
          (FLOAT32* __restrict__)a.const_data_ptr<float>();

      WORD32 p_out_shape[kNnlibMaxDim];
      WORD32 p_inp1_shape[kNnlibMaxDim];

      for (int i = 0; i < out_dim; i++)
        p_out_shape[i] = out.size(i);
      for (int i = 0; i < a_dim; i++)
        p_inp1_shape[i] = a.size(i);

      xa_nn_broadcast_32_32(
          (WORD32*)ptr1, p_out_shape, (WORD32*)pin1, p_inp1_shape, out_dim);

      FLOAT32* __restrict__ p_out =
          (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp1 = (const FLOAT32* __restrict__)ptr1;
      const FLOAT32* __restrict__ p_inp2 =
          (const FLOAT32* __restrict__)b.const_data_ptr<float>();

      xa_nn_elm_pow_f32(p_out, p_inp1, p_inp2, num_elm);

    } else if (b_is_broadcasted && (!a_is_broadcasted)) {
      WORD32* __restrict__ ptr1 =
          (WORD32* __restrict__)kernels::allocate_temp_memory(
              ctx, num_elm * sizeof(int));

      ET_KERNEL_CHECK(ctx, ptr1 != nullptr, MemoryAllocationFailed, out);

      WORD32* __restrict__ pin1 =
          (WORD32* __restrict__)b.const_data_ptr<float>();

      WORD32 p_out_shape[kNnlibMaxDim];
      WORD32 p_inp1_shape[kNnlibMaxDim];

      for (int i = 0; i < out_dim; i++)
        p_out_shape[i] = out.size(i);
      for (int i = 0; i < b_dim; i++)
        p_inp1_shape[i] = b.size(i);

      xa_nn_broadcast_32_32(ptr1, p_out_shape, pin1, p_inp1_shape, out_dim);

      FLOAT32* __restrict__ p_out =
          (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp1 =
          (const FLOAT32* __restrict__)a.const_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp2 = (const FLOAT32* __restrict__)ptr1;

      xa_nn_elm_pow_f32(p_out, p_inp1, p_inp2, num_elm);

    } else {
      FLOAT32* __restrict__ p_out =
          (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp1 =
          (const FLOAT32* __restrict__)a.const_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp2 =
          (const FLOAT32* __restrict__)b.const_data_ptr<float>();

      xa_nn_elm_pow_f32(p_out, p_inp1, p_inp2, num_elm);
    }
    return out;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "pow.Tensor_Tensor_out";

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    apply_bitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [](const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
          return std::pow(val_a, val_b);
        },
        ctx,
        a,
        SupportedTensorDtypes::REALHBBF16,
        b,
        SupportedTensorDtypes::REALHBBF16,
        out,
        SupportedTensorDtypes::REALHBF16);
  });

  return out;
}

Tensor& pow_Tensor_Scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promote_type_with_scalar(a.scalar_type(), b);

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (canCast(common_type, out.scalar_type()) &&
       common_type != ScalarType::Bool),
      InvalidArgument,
      out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = get_compute_type(common_type);
  if (compute_type != ScalarType::Float) {
    compute_type = ScalarType::Double;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "pow.Tensor_Scalar_out";

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_b = scalar_to<CTYPE_COMPUTE>(b);
    apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [val_b](const CTYPE_COMPUTE val_a) { return std::pow(val_a, val_b); },
        ctx,
        a,
        SupportedTensorDtypes::REALHBBF16,
        out,
        SupportedTensorDtypes::REALHBF16);
  });

  return out;
}

Tensor& pow_Scalar_out(
    KernelRuntimeContext& ctx,
    const Scalar& a,
    const Tensor& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promote_type_with_scalar(b.scalar_type(), a);

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (canCast(common_type, out.scalar_type()) &&
       common_type != ScalarType::Bool),
      InvalidArgument,
      out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(b, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, b.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = get_compute_type(common_type);
  if (compute_type != ScalarType::Float) {
    compute_type = ScalarType::Double;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "pow.Scalar_out";

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_a = scalar_to<CTYPE_COMPUTE>(a);
    apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [val_a](const CTYPE_COMPUTE val_b) { return std::pow(val_a, val_b); },
        ctx,
        b,
        SupportedTensorDtypes::REALHBBF16,
        out,
        SupportedTensorDtypes::REALHBF16);
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
