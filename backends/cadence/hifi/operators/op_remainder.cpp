/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include "kernels.h"

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::aten::RuntimeContext;
using executorch::runtime::can_cast;
using executorch::runtime::canCast;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::promoteTypes;
using torch::executor::apply_binary_elementwise_fn;
using torch::executor::apply_unary_map_fn;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;
using torch::executor::native::utils::extract_scalar;
using torch::executor::native::utils::get_scalar_dtype;
using torch::executor::native::utils::promote_type_with_scalar;
using torch::executor::native::utils::remainder_override;

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
  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  ET_SWITCH_REAL_TYPES_AND(
      Bool, a_type, ctx, "remainder.Tensor_out", CTYPE_A, [&]() {
        ET_SWITCH_REAL_TYPES_AND(
            Bool, b_type, ctx, "remainder.Tensor_out", CTYPE_B, [&]() {
              using CTYPE_IN = typename torch::executor::
                  promote_types<CTYPE_A, CTYPE_B>::type;
              ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
              ET_SWITCH_REAL_TYPES(
                  out_type, ctx, "remainder.Tensor_out", CTYPE_OUT, [&]() {
                    RemainderInner<
                        can_cast<CTYPE_IN, CTYPE_OUT>::value,
                        CTYPE_A,
                        CTYPE_B,
                        CTYPE_IN,
                        CTYPE_OUT>::run(a, b, out);
                  });
            });
      });

  return out;
}

Tensor& remainder_Scalar_out(
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

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = get_scalar_dtype(b);
  ScalarType common_type = promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  ET_SWITCH_REAL_TYPES_AND(
      Bool, a_type, ctx, "remainder.Scalar_out", CTYPE_A, [&]() {
        ET_SWITCH_SCALAR_OBJ_TYPES(
            b_type, ctx, "remainder.Scalar_out", CTYPE_B, [&]() {
              CTYPE_B val_b = 0;
              extract_scalar(b, &val_b);
              ET_SWITCH_REAL_TYPES(
                  common_type, ctx, "remainder.Scalar_out", CTYPE_IN, [&]() {
                    ET_SWITCH_REAL_TYPES(
                        out_type,
                        ctx,
                        "remainder.Scalar_out",
                        CTYPE_OUT,
                        [&]() {
                          apply_unary_map_fn(
                              [val_b](const CTYPE_A val_a) {
                                CTYPE_IN a_casted =
                                    static_cast<CTYPE_IN>(val_a);
                                CTYPE_IN b_casted =
                                    static_cast<CTYPE_IN>(val_b);
                                CTYPE_IN value =
                                    remainder_override(a_casted, b_casted);

                                return static_cast<CTYPE_OUT>(value);
                              },
                              a.const_data_ptr<CTYPE_A>(),
                              out.mutable_data_ptr<CTYPE_OUT>(),
                              out.numel());
                        });
                  });
            });
      });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
