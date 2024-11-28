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
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::can_cast;
using executorch::runtime::canCast;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::promoteTypes;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;

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
struct PowInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct PowInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void run(const Tensor& a, const Tensor& b, Tensor& out) {
    torch::executor::apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = std::pow(a_casted, b_casted);
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
struct PowInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug {};

} // namespace

Tensor& pow_Tensor_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type, /*half_to_float*/ true);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(
      ctx, common_type != exec_aten::ScalarType::Bool, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  constexpr auto name = "pow.Tensor_Tensor_out";
  constexpr int kNnlibMaxDim = 16;
  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  bool optimized = true;

  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted && b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  if (out_type != ScalarType::Float)
    optimized = false;

  if (max_dim > kNnlibMaxDim)
    optimized = false;

  WORD32 num_elm = out.numel();

  if (optimized) {
    if (broadcast) {
      WORD32* __restrict__ ptr1 =
          (WORD32* __restrict__)malloc(num_elm * sizeof(WORD32));
      WORD32* __restrict__ ptr2 =
          (WORD32* __restrict__)malloc(num_elm * sizeof(WORD32));

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

      free(ptr1);
      free(ptr2);
    } else if (a_is_broadcasted && (!b_is_broadcasted)) {
      FLOAT32* __restrict__ ptr1 =
          (FLOAT32* __restrict__)malloc((num_elm + 2) * sizeof(WORD32));

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

      free(ptr1);
    } else if (b_is_broadcasted && (!a_is_broadcasted)) {
      WORD32* __restrict__ ptr1 =
          (WORD32* __restrict__)malloc(num_elm * sizeof(WORD32));

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

      free(ptr1);
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

  ET_SWITCH_REALHB_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      using CTYPE_IN = typename torch::executor::
          promote_types<CTYPE_A, CTYPE_B, /*half_to_float*/ true>::type;
      ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
      ET_SWITCH_REALH_TYPES(out_type, ctx, name, CTYPE_OUT, [&]() {
        PowInner<
            !std::is_same<CTYPE_IN, bool>::value &&
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

Tensor& pow_Tensor_Scalar_out(
    KernelRuntimeContext& ctx,
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
  ScalarType b_type = torch::executor::native::utils::get_scalar_dtype(b);
  ScalarType common_type =
      torch::executor::native::utils::promote_type_with_scalar(
          a_type, b, /*half_to_float*/ false);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  constexpr auto name = "pow.Tensor_Scalar_out";
  if (common_type == ScalarType::Half) {
    common_type = ScalarType::Float;
  }

  ET_SWITCH_REALHB_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES(common_type, ctx, name, CTYPE_IN, [&]() {
        ET_SWITCH_REALH_TYPES(out_type, ctx, name, CTYPE_OUT, [&]() {
          CTYPE_B val_b = 0;
          torch::executor::native::utils::extract_scalar(b, &val_b);
          torch::executor::apply_unary_map_fn(
              [val_b](const CTYPE_A val_a) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                CTYPE_IN value = std::pow(a_casted, b_casted);

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

Tensor& pow_Scalar_out(
    KernelRuntimeContext& ctx,
    const Scalar& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, b.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType a_type = torch::executor::native::utils::get_scalar_dtype(a);
  ScalarType b_type = b.scalar_type();
  ScalarType common_type =
      torch::executor::native::utils::promote_type_with_scalar(
          b_type, a, /*half_to_float*/ false);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  constexpr auto name = "pow.Scalar_out";
  if (common_type == ScalarType::Half) {
    common_type = ScalarType::Float;
  }

  ET_SWITCH_SCALAR_OBJ_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES(common_type, ctx, name, CTYPE_IN, [&]() {
        ET_SWITCH_REALH_TYPES(out_type, ctx, name, CTYPE_OUT, [&]() {
          CTYPE_A val_a = 0;
          torch::executor::native::utils::extract_scalar(a, &val_a);

          torch::executor::apply_unary_map_fn(
              [val_a](const CTYPE_B val_b) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                CTYPE_IN value = std::pow(a_casted, b_casted);
                return static_cast<CTYPE_OUT>(value);
              },
              b.const_data_ptr<CTYPE_B>(),
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
