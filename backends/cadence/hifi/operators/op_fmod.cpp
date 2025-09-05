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
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::aten::RuntimeContext;
using executorch::runtime::can_cast;
using executorch::runtime::canCast;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::is_integral_type;
using executorch::runtime::isIntegralType;
using executorch::runtime::promoteTypes;
using torch::executor::apply_binary_elementwise_fn;
using torch::executor::apply_unary_map_fn;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;
using torch::executor::native::utils::extract_scalar;
using torch::executor::native::utils::get_scalar_dtype;
using torch::executor::native::utils::promote_type_with_scalar;

namespace cadence {
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
struct FmodInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct FmodInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void
  run(const Tensor& a, const Tensor& b, Tensor& out, bool& div_by_zero_error) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [&div_by_zero_error](const CTYPE_A val_a, const CTYPE_B val_b) {
          if (is_integral_type<CTYPE_IN, /*includeBool=*/true>::value) {
            if (val_b == 0) {
              div_by_zero_error = true;
              return static_cast<CTYPE_OUT>(0);
            }
          }
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = std::fmod(a_casted, b_casted);

          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, Tensor&, bool&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct FmodInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug {};

} // namespace

Tensor& fmod_Tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  static constexpr const char op_name[] = "fmod.Tensor_out";
  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */

  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  bool optimized = true;
  /*find broadcast*/
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted || b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  if ((a.scalar_type() == ScalarType::Float) ||
      (b.scalar_type() == ScalarType::Float))
    optimized = false;

  if ((a_dim == 0) || (b_dim == 0))
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

      WORD32 val = xa_nn_elm_fmod_broadcast_4D_f32xf32_f32(
          p_out, p_out_shape, p_inp1, p_inp1_shape, p_inp2, p_inp2_shape);
    } else {
      WORD32 num_elm = out.numel();

      WORD32 val = xa_nn_elm_fmod_f32xf32_f32(p_out, p_inp1, p_inp2, num_elm);
    }

    return out;
  }

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  auto div_by_zero_error = false;

  // Compute Dtype
  ScalarType compute_type =
      torch::executor::native::utils::get_compute_type(common_type);
  if (compute_type != ScalarType::Float) {
    compute_type = ScalarType::Double;
  }
  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    torch::executor::native::utils::apply_bitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        torch::executor::native::utils::SupportedTensorDtypes::REALHBF16>(
        [&div_by_zero_error](
            const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
          // TODO: rewrite this to be vectorization-capable?
          CTYPE_COMPUTE value = 0;
          if (is_integral_type<CTYPE_COMPUTE, /*includeBool=*/true>::value) {
            if (val_b == 0) {
              div_by_zero_error = true;
              return value;
            }
          }
          value = std::fmod(val_a, val_b);
          return value;
        },
        ctx,
        a,
        torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
        b,
        torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  ET_KERNEL_CHECK_MSG(
      ctx,
      !div_by_zero_error,
      InvalidArgument,
      out,
      "Fmod operation encountered integer division by zero");

  return out;
}

Tensor& fmod_Scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "fmod.Scalar_out";

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = get_scalar_dtype(b);
  ScalarType common_type = promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  // Check for integer division by zero
  if (isIntegralType(common_type, /*includeBool=*/true)) {
    auto is_zero = false;
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, op_name, CTYPE_B, [&]() {
      CTYPE_B val_b = 0;
      extract_scalar(b, &val_b);
      is_zero = (val_b == 0);
    });

    ET_KERNEL_CHECK_MSG(
        ctx,
        !is_zero,
        InvalidArgument,
        out,
        "Fmod operation encountered integer division by zero");
  }
  // Compute Dtype
  ScalarType compute_type =
      torch::executor::native::utils::get_compute_type(common_type);
  if (compute_type != ScalarType::Float) {
    compute_type = ScalarType::Double;
  }

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_b =
        torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(b);
    torch::executor::native::utils::apply_unitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        torch::executor::native::utils::SupportedTensorDtypes::REALHBF16>(
        [val_b](const auto val_a) {
          return executorch::math::fmod(val_a, (decltype(val_a))val_b);
        },
        ctx,
        a,
        torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });
  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
