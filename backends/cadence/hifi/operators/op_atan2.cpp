/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>

using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

Tensor& atan2_out(
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

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  constexpr auto name = "atan2.out";
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
          (WORD32* __restrict__)kernels::allocate_temp_memory(
              ctx, num_elm * sizeof(WORD32));

      ET_KERNEL_CHECK(ctx, ptr1 != nullptr, MemoryAllocationFailed, out);

      WORD32* __restrict__ ptr2 =
          (WORD32* __restrict__)kernels::allocate_temp_memory(
              ctx, num_elm * sizeof(WORD32));

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

      WORD32 ret_val =
          xa_nn_broadcast_32_32(ptr1, p_out_shape, pin1, p_inp1_shape, out_dim);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

      ret_val =
          xa_nn_broadcast_32_32(ptr2, p_out_shape, pin2, p_inp2_shape, out_dim);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

      FLOAT32* __restrict__ p_out =
          (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp1 = (const FLOAT32* __restrict__)ptr1;
      const FLOAT32* __restrict__ p_inp2 = (const FLOAT32* __restrict__)ptr2;

      xa_nn_elm_atan2_f32(p_out, p_inp1, p_inp2, num_elm);

    } else if (a_is_broadcasted && (!b_is_broadcasted)) {
      FLOAT32* __restrict__ ptr1 =
          (FLOAT32* __restrict__)kernels::allocate_temp_memory(
              ctx, num_elm * sizeof(WORD32));

      ET_KERNEL_CHECK(ctx, ptr1 != nullptr, MemoryAllocationFailed, out);

      FLOAT32* __restrict__ pin1 =
          (FLOAT32* __restrict__)a.const_data_ptr<float>();

      WORD32 p_out_shape[kNnlibMaxDim];
      WORD32 p_inp1_shape[kNnlibMaxDim];

      for (int i = 0; i < out_dim; i++)
        p_out_shape[i] = out.size(i);
      for (int i = 0; i < a_dim; i++)
        p_inp1_shape[i] = a.size(i);

      WORD32 ret_val = xa_nn_broadcast_32_32(
          (WORD32*)ptr1, p_out_shape, (WORD32*)pin1, p_inp1_shape, out_dim);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

      FLOAT32* __restrict__ p_out =
          (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp1 = (const FLOAT32* __restrict__)ptr1;
      const FLOAT32* __restrict__ p_inp2 =
          (const FLOAT32* __restrict__)b.const_data_ptr<float>();

      xa_nn_elm_atan2_f32(p_out, p_inp1, p_inp2, num_elm);

    } else if (b_is_broadcasted && (!a_is_broadcasted)) {
      WORD32* __restrict__ ptr1 =
          (WORD32* __restrict__)kernels::allocate_temp_memory(
              ctx, num_elm * sizeof(WORD32));

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

      xa_nn_elm_atan2_f32(p_out, p_inp1, p_inp2, num_elm);

    } else {
      FLOAT32* __restrict__ p_out =
          (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp1 =
          (const FLOAT32* __restrict__)a.const_data_ptr<float>();
      const FLOAT32* __restrict__ p_inp2 =
          (const FLOAT32* __restrict__)b.const_data_ptr<float>();

      xa_nn_elm_atan2_f32(p_out, p_inp1, p_inp2, num_elm);
    }
    return out;
  }

  ET_SWITCH_REALHB_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      ET_SWITCH_FLOATH_TYPES(out_type, ctx, name, CTYPE_OUT, [&]() {
        torch::executor::
            apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
                [](const CTYPE_A val_a, const CTYPE_B val_b) {
                  CTYPE_OUT casted_a = static_cast<CTYPE_OUT>(val_a);
                  CTYPE_OUT casted_b = static_cast<CTYPE_OUT>(val_b);
                  return static_cast<CTYPE_OUT>(std::atan2(casted_a, casted_b));
                },
                a,
                b,
                out);
      });
    });
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence