/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/comparison_op.h>
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

namespace impl {
namespace HiFi {
namespace native {

Tensor& eq_Tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType out_type = out.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char name[] = "eq.Tensor_out";
  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */

  int a_dim = a.dim();
  int b_dim = b.dim();
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
          p_out, out_shape, p_inp1, inp1_shape, p_inp2, inp2_shape, 4);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
    } else {
      int num_elm = out.numel();

      WORD32 ret_val = xa_nn_elm_greater_lesser_equal_f32xf32_f32(
          p_out, p_inp1, p_inp2, num_elm, 4);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
    }

    return out;
  }

  return torch::executor::native::internal::
      comparison_tensor_out<std::equal_to, name>(ctx, a, b, out);
}

Tensor& eq_Scalar_out(
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
  static constexpr const char name[] = "eq.Scalar_out";

  return torch::executor::native::internal::
      comparison_scalar_out<std::equal_to, name>(ctx, a, b, out);
}

} // namespace native
} // namespace HiFi
} // namespace impl
