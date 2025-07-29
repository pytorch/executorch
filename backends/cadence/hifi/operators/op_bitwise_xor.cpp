/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// patternlint-disable-next-line executorch-cpp-nostdinc
#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/pattern/bitwise_op.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::can_cast;
using executorch::runtime::canCast;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::promoteTypes;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

Tensor& bitwise_xor_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  static constexpr const char op_name[] = "bitwise_xor.Tensor_out";
  constexpr int kNnlibMaxDim = 16;
  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  bool optimized = true;

  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted && b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  if (out_type != ScalarType::Bool)
    optimized = false;

  if (max_dim > kNnlibMaxDim)
    optimized = false;

  WORD32 num_elm = out.numel();

  if (optimized) {
    if (broadcast) {
      WORD8* __restrict__ ptr1 =
          (WORD8* __restrict__)kernels::allocate_temp_memory(ctx, num_elm);
      WORD8* __restrict__ ptr2 =
          (WORD8* __restrict__)kernels::allocate_temp_memory(ctx, num_elm);

      const WORD8* __restrict__ pin1 =
          (const WORD8* __restrict__)a.const_data_ptr<bool>();
      const WORD8* __restrict__ pin2 =
          (const WORD8* __restrict__)b.const_data_ptr<bool>();

      WORD8* __restrict__ p_out =
          (WORD8* __restrict__)out.mutable_data_ptr<bool>();

      WORD32 p_out_shape[kNnlibMaxDim];
      WORD32 p_inp2_shape[kNnlibMaxDim];
      WORD32 p_inp1_shape[kNnlibMaxDim];

      for (int i = 0; i < out_dim; i++)
        p_out_shape[i] = out.size(i);
      for (int i = 0; i < a_dim; i++)
        p_inp1_shape[i] = a.size(i);
      for (int i = 0; i < b_dim; i++)
        p_inp2_shape[i] = b.size(i);

      xa_nn_broadcast_8_8(ptr1, p_out_shape, pin1, p_inp1_shape, out_dim);

      xa_nn_broadcast_8_8(ptr2, p_out_shape, pin2, p_inp2_shape, out_dim);

      const WORD8* __restrict__ p_inp1 = (const WORD8* __restrict__)ptr1;
      const WORD8* __restrict__ p_inp2 = (const WORD8* __restrict__)ptr2;

      xa_nn_elm_logicalxor_boolxbool_bool(p_out, p_inp1, p_inp2, num_elm);
    } else if (a_is_broadcasted && !b_is_broadcasted) {
      WORD8* __restrict__ ptr1 =
          (WORD8* __restrict__)kernels::allocate_temp_memory(ctx, num_elm);

      const WORD8* __restrict__ pin1 =
          (const WORD8* __restrict__)a.const_data_ptr<bool>();
      const WORD8* __restrict__ p_inp2 =
          (const WORD8* __restrict__)b.const_data_ptr<bool>();

      WORD8* __restrict__ p_out =
          (WORD8* __restrict__)out.mutable_data_ptr<bool>();

      WORD32 p_out_shape[kNnlibMaxDim];
      WORD32 p_inp1_shape[kNnlibMaxDim];

      for (int i = 0; i < out_dim; i++)
        p_out_shape[i] = out.size(i);
      for (int i = 0; i < a_dim; i++)
        p_inp1_shape[i] = a.size(i);

      xa_nn_broadcast_8_8(ptr1, p_out_shape, pin1, p_inp1_shape, out_dim);

      const WORD8* __restrict__ p_inp1 = (const WORD8* __restrict__)ptr1;

      xa_nn_elm_logicalxor_boolxbool_bool(p_out, p_inp1, p_inp2, num_elm);
    } else if (!a_is_broadcasted && b_is_broadcasted) {
      WORD8* __restrict__ ptr1 =
          (WORD8* __restrict__)kernels::allocate_temp_memory(ctx, num_elm);

      const WORD8* __restrict__ p_inp1 =
          (const WORD8* __restrict__)a.const_data_ptr<bool>();
      const WORD8* __restrict__ pinp2 =
          (const WORD8* __restrict__)b.const_data_ptr<bool>();

      WORD8* __restrict__ p_out =
          (WORD8* __restrict__)out.mutable_data_ptr<bool>();

      WORD32 p_out_shape[kNnlibMaxDim];
      WORD32 p_inp2_shape[kNnlibMaxDim];

      for (int i = 0; i < out_dim; i++)
        p_out_shape[i] = out.size(i);
      for (int i = 0; i < b_dim; i++)
        p_inp2_shape[i] = b.size(i);

      xa_nn_broadcast_8_8(ptr1, p_out_shape, pinp2, p_inp2_shape, out_dim);

      const WORD8* __restrict__ p_inp2 = (const WORD8* __restrict__)ptr1;

      xa_nn_elm_logicalxor_boolxbool_bool(p_out, p_inp1, p_inp2, num_elm);
    } else {
      const WORD8* __restrict__ p_inp1 =
          (const WORD8* __restrict__)a.const_data_ptr<bool>();
      const WORD8* __restrict__ p_inp2 =
          (const WORD8* __restrict__)b.const_data_ptr<bool>();

      WORD8* __restrict__ p_out =
          (WORD8* __restrict__)out.mutable_data_ptr<bool>();

      xa_nn_elm_logicalxor_boolxbool_bool(p_out, p_inp1, p_inp2, num_elm);
    }
    return out;
  }

  return torch::executor::native::internal::
      bitwise_tensor_out<std::bit_xor, op_name>(ctx, a, b, out);
}

Tensor& bitwise_xor_Scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "bitwise_xor.Scalar_out";
  return torch::executor::native::internal::
      bitwise_scalar_out<std::bit_xor, op_name>(ctx, a, b, out);
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
