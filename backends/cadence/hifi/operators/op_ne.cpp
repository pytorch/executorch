/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::aten::RuntimeContext;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::promoteTypes;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

Tensor& ne_tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  constexpr auto name = "ne.Tensor_out";
  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */

  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  bool optimized = true;
  /*find broadcast*/
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted || b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  if (out_type != ScalarType::Float)
    optimized = false;

  if ((a_dim == 0) || (b_dim == 0))
    optimized = false;

  if ((broadcast == 1) && (max_dim > kNnlibMaxDim))
    optimized = false;

  if (optimized) {
    int8_t* __restrict__ p_out =
        (int8_t* __restrict__)out.mutable_data_ptr<int8_t>();
    const float* __restrict__ p_inp1 =
        (const float* __restrict__)a.const_data_ptr<float>();
    const float* __restrict__ p_inp2 =
        (const float* __restrict__)b.const_data_ptr<float>();

    if (broadcast) {
      int out_shape[kNnlibMaxDim];
      int inp1_shape[kNnlibMaxDim];
      int inp2_shape[kNnlibMaxDim];

      for (int i = 0; i < kNnlibMaxDim; i++) {
        inp1_shape[i] = 1;
        inp2_shape[i] = 1;
        out_shape[i] = 1;
      }

      int off_o = kNnlibMaxDim - out.dim();
      int off_a = kNnlibMaxDim - a.dim();
      int off_b = kNnlibMaxDim - b.dim();

      for (int i = 0; i < out.dim(); i++)
        out_shape[i + off_o] = out.size(i);
      for (int i = 0; i < a.dim(); i++)
        inp1_shape[i + off_a] = a.size(i);
      for (int i = 0; i < b.dim(); i++)
        inp2_shape[i + off_b] = b.size(i);

      WORD32 ret_val = xa_nn_elm_greater_lesser_equal_broadcast_4D_f32xf32_f32(
          p_out, out_shape, p_inp1, inp1_shape, p_inp2, inp2_shape, 5);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
    } else {
      int num_elm = out.numel();

      WORD32 ret_val = xa_nn_elm_greater_lesser_equal_f32xf32_f32(
          p_out, p_inp1, p_inp2, num_elm, 5);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
    }

    return out;
  }

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, name, CTYPE_B, [&]() {
      using CTYPE_IN =
          typename torch::executor::promote_types<CTYPE_A, CTYPE_B>::type;
      ET_DCHECK(
          CppTypeToScalarType<CTYPE_IN>::value == promoteTypes(a_type, b_type));
      ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, name, CTYPE_OUT, [&]() {
        torch::executor::
            apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
                [](const CTYPE_A val_a, const CTYPE_B val_b) {
                  CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                  CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                  bool value = a_casted != b_casted;
                  return static_cast<CTYPE_OUT>(value);
                },
                a,
                b,
                out);
      });
    });
  });

  return out;
}

Tensor& ne_scalar_out(
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

  constexpr auto name = "ne.Scalar_out";

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = torch::executor::native::utils::get_scalar_dtype(b);
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      using CTYPE_IN =
          typename torch::executor::promote_types<CTYPE_A, CTYPE_B>::type;
      ET_DCHECK(
          CppTypeToScalarType<CTYPE_IN>::value == promoteTypes(a_type, b_type));
      ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, name, CTYPE_OUT, [&]() {
        CTYPE_B val_b = 0;
        torch::executor::native::utils::extract_scalar(b, &val_b);
        torch::executor::apply_unary_map_fn(
            [val_b](const CTYPE_A val_a) {
              CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
              CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
              bool value = a_casted != b_casted;
              return static_cast<CTYPE_OUT>(value);
            },
            a.const_data_ptr<CTYPE_A>(),
            out.mutable_data_ptr<CTYPE_OUT>(),
            out.numel());
      });
    });
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence