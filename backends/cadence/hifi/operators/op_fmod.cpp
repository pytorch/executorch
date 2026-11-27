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
using executorch::runtime::is_integral_type;
using executorch::runtime::promoteTypes;
using torch::executor::apply_binary_elementwise_fn;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;
using torch::executor::native::utils::promote_type_with_scalar;

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

// Computes integers as integers so that int64 values above 2^53 are not
// rounded through double.
template <typename CTYPE>
CTYPE fmod_value(CTYPE a, CTYPE b) {
  if constexpr (std::is_integral_v<CTYPE>) {
    // a % -1 overflows for the most negative value; the result is always 0.
    if constexpr (std::is_signed_v<CTYPE>) {
      if (b == -1) {
        return 0;
      }
    }
    return static_cast<CTYPE>(a % b);
  } else {
    return std::fmod(a, b);
  }
}

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

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  auto div_by_zero_error = false;

  // Compute Dtype
  ScalarType compute_type =
      torch::executor::native::utils::get_compute_type(common_type);
  ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
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
          value = fmod_value(val_a, val_b);
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
  ScalarType common_type = promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type =
      torch::executor::native::utils::get_compute_type(common_type);

  bool div_by_zero_error = false;

  ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    // Like ATen, the scalar takes the compute dtype first, so a scalar such as
    // 256 for an int8 tensor becomes 0.
    const CTYPE_COMPUTE val_b =
        torch::executor::native::utils::scalar_to<CTYPE_COMPUTE>(b);
    if (is_integral_type<CTYPE_COMPUTE, /*includeBool=*/true>::value &&
        val_b == 0 && a.numel() > 0) {
      div_by_zero_error = true;
      return;
    }
    torch::executor::native::utils::apply_unitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        torch::executor::native::utils::SupportedTensorDtypes::REALHBF16>(
        [val_b](const CTYPE_COMPUTE val_a) { return fmod_value(val_a, val_b); },
        ctx,
        a,
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

} // namespace native
} // namespace HiFi
} // namespace impl
