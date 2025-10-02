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
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::RuntimeContext;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::can_cast;
using executorch::runtime::canCast;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::is_integral_type;
using executorch::runtime::promoteTypes;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::apply_binary_elementwise_fn;
using torch::executor::apply_unary_map_fn;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;
using torch::executor::native::utils::apply_bitensor_elementwise_fn;
using torch::executor::native::utils::apply_unitensor_elementwise_fn;
using torch::executor::native::utils::extract_scalar;
using torch::executor::native::utils::get_compute_type;
using torch::executor::native::utils::get_scalar_dtype;
using torch::executor::native::utils::promote_type_with_scalar;
using torch::executor::native::utils::remainder_override;
using torch::executor::native::utils::scalar_to;
using torch::executor::native::utils::SupportedTensorDtypes;

namespace impl {
namespace HiFi {
namespace native {

namespace {
template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct RemainderInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct RemainderInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void run(const Tensor& a, const Tensor& b, Tensor& out) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = remainder_override(a_casted, b_casted);

          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, Tensor&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct RemainderInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug {};

} // namespace
Tensor& remainder_Tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

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
  constexpr int kNnlibMaxDim =
      4; /*fallback to not optimised if broadcast and dim > 4 */

  bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  bool broadcast = (a_is_broadcasted || b_is_broadcasted);

  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  bool optimized = true;

  if ((a.scalar_type() != ScalarType::Float) ||
      (b.scalar_type() != ScalarType::Float))
    optimized = false;

  if ((broadcast == true) && (max_dim > kNnlibMaxDim))
    optimized = false;

  if (optimized) {
    FLOAT32* __restrict__ p_out =
        (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
    const FLOAT32* __restrict__ p_inp1 =
        (const FLOAT32* __restrict__)a.const_data_ptr<float>();
    const FLOAT32* __restrict__ p_inp2 =
        (const FLOAT32* __restrict__)b.const_data_ptr<float>();

    if (broadcast) {
      WORD32 p_out_shape[kNnlibMaxDim];
      WORD32 p_inp1_shape[kNnlibMaxDim];
      WORD32 p_inp2_shape[kNnlibMaxDim];

      for (int i = 0; i < kNnlibMaxDim; i++) {
        p_inp1_shape[i] = 1;
        p_inp2_shape[i] = 1;
        p_out_shape[i] = 1;
      }

      int off_o = kNnlibMaxDim - out.dim();
      int off_a = kNnlibMaxDim - a.dim();
      int off_b = kNnlibMaxDim - b.dim();

      for (int i = 0; i < out.dim(); i++)
        p_out_shape[i + off_o] = out.size(i);
      for (int i = 0; i < a.dim(); i++)
        p_inp1_shape[i + off_a] = a.size(i);
      for (int i = 0; i < b.dim(); i++)
        p_inp2_shape[i + off_b] = b.size(i);

      WORD32 ret_val = xa_nn_elm_remainder_broadcast_4D_f32xf32_f32(
          p_out, p_out_shape, p_inp1, p_inp1_shape, p_inp2, p_inp2_shape);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
    } else {
      WORD32 ret_val =
          xa_nn_elm_remainder_f32xf32_f32(p_out, p_inp1, p_inp2, out.numel());

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
    }
    return out;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "remainder.Tensor_out";

  bool div_by_zero_error = false;

  ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    apply_bitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [&div_by_zero_error](
            const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
          CTYPE_COMPUTE value = 0;
          if (is_integral_type<CTYPE_COMPUTE, /*includeBool=*/true>::value) {
            if (val_b == 0) {
              div_by_zero_error = true;
              return value;
            }
          }
          value = remainder_override(val_a, val_b);
          return value;
        },
        ctx,
        a,
        SupportedTensorDtypes::REALHBBF16,
        b,
        SupportedTensorDtypes::REALHBBF16,
        out,
        SupportedTensorDtypes::REALHBF16);
  });

  ET_KERNEL_CHECK_MSG(
      ctx,
      !div_by_zero_error,
      InvalidArgument,
      out,
      "Remainder operation encountered integer division by zero");

  return out;
}

Tensor& remainder_Scalar_out(
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

  // Check for intergral division by zero
  ET_KERNEL_CHECK_MSG(
      ctx,
      !(executorch::runtime::isIntegralType(common_type, true) &&
        scalar_to<double>(b) == 0),
      InvalidArgument,
      out,
      "Remainder operation encountered integer division by zero");

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "remainder.Scalar_out";

  ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_b = scalar_to<CTYPE_COMPUTE>(b);
    apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [val_b](const CTYPE_COMPUTE val_a) {
          return remainder_override(val_a, val_b);
        },
        ctx,
        a,
        SupportedTensorDtypes::REALHBBF16,
        out,
        SupportedTensorDtypes::REALHBF16);
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
