/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::RuntimeContext;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::promoteTypes;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;
using torch::executor::native::utils::apply_tritensor_elementwise_fn;
using torch::executor::native::utils::get_compute_type;
using torch::executor::native::utils::SupportedTensorDtypes;

namespace impl {
namespace HiFi {
namespace native {

Tensor& where_self_out(
    RuntimeContext& ctx,
    const Tensor& cond,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promoteTypes(a.scalar_type(), b.scalar_type());

  // Check Common Dtype
  ET_KERNEL_CHECK(ctx, common_type == out.scalar_type(), InvalidArgument, out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(cond, a, b, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, cond, out) == Error::Ok,
      InvalidArgument,
      out);

  // Compute Dtype
  ScalarType compute_type = get_compute_type(common_type);

  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */

  int a_dim = a.dim(), b_dim = b.dim(), con_dim = cond.dim(),
      out_dim = out.dim();
  bool optimized = 1;
  /*find broadcast*/
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool cond_is_broadcasted = !out.sizes().equals(cond.sizes());
  const bool broadcast =
      (a_is_broadcasted || b_is_broadcasted || cond_is_broadcasted);

  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = cond.dim() > max_dim ? cond.dim() : max_dim;
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();

  if ((a_type != ScalarType::Float) || (b_type != ScalarType::Float))
    optimized = 0;

  if ((a_dim == 0) || (b_dim == 0) || (con_dim == 0))
    optimized = 0;

  if ((broadcast == 1) && (max_dim > kNnlibMaxDim))
    optimized = 0;

  if (optimized) {
    const float* a_data = a.const_data_ptr<float>();
    const float* b_data = b.const_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();
    const unsigned char* con = cond.const_data_ptr<uint8_t>();

    if (broadcast == 1) {
      int out_shape[kNnlibMaxDim];
      int inp1_shape[kNnlibMaxDim];
      int inp2_shape[kNnlibMaxDim];
      int con_shape[kNnlibMaxDim];

      for (int i = 0; i < kNnlibMaxDim; i++) {
        con_shape[i] = 1;
        out_shape[i] = 1;
        inp1_shape[i] = 1;
        inp2_shape[i] = 1;
      }

      int off_o = kNnlibMaxDim - out.dim();
      int off_a = kNnlibMaxDim - a.dim();
      int off_b = kNnlibMaxDim - b.dim();
      int off_c = kNnlibMaxDim - cond.dim();

      for (int i = 0; i < out.dim(); i++)
        out_shape[i + off_o] = out.size(i);
      for (int i = 0; i < a.dim(); i++)
        inp1_shape[i + off_a] = a.size(i);
      for (int i = 0; i < b.dim(); i++)
        inp2_shape[i + off_b] = b.size(i);
      for (int i = 0; i < cond.dim(); i++)
        con_shape[i + off_c] = cond.size(i);

      if (con_shape[0] != out_shape[0] || con_shape[1] != out_shape[1] ||
          con_shape[2] != out_shape[2] || con_shape[3] != out_shape[3]) {
        void* p_scratch = (void*)kernels::allocate_temp_memory(
            ctx,
            (out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]) *
                sizeof(int));

        ET_KERNEL_CHECK(ctx, p_scratch != nullptr, MemoryAllocationFailed, out);

        const unsigned char* p_brd_cond = (const unsigned char*)p_scratch;
        xa_nn_broadcast_8_8(
            (WORD8* __restrict__)p_brd_cond,
            out_shape,
            (const WORD8* __restrict__)con,
            con_shape,
            4);

        for (int i = 0; i < 4; i++) {
          con_shape[i] = out_shape[i];
        }
        xa_nn_elm_where_broadcast_4D_f32xf32_f32(
            out_data,
            out_shape,
            a_data,
            inp1_shape,
            b_data,
            inp2_shape,
            p_brd_cond,
            con_shape);

      } else {
        xa_nn_elm_where_broadcast_4D_f32xf32_f32(
            out_data,
            out_shape,
            a_data,
            inp1_shape,
            b_data,
            inp2_shape,
            con,
            con_shape);
      }
    } else {
      xa_nn_elm_where_f32xf32_f32(out_data, a_data, b_data, con, out.numel());
    }
    return out;
  }
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "where.self_out";

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    apply_tritensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [](const CTYPE_COMPUTE val_a,
           const CTYPE_COMPUTE val_b,
           const CTYPE_COMPUTE val_c) { return val_c ? val_a : val_b; },
        ctx,
        a,
        SupportedTensorDtypes::REALHBBF16,
        b,
        SupportedTensorDtypes::REALHBBF16,
        cond,
        SupportedTensorDtypes::BOOL_OR_BYTE,
        out,
        SupportedTensorDtypes::SAME_AS_COMMON);
  });

  return out;
}

Tensor& where_out(
    RuntimeContext& ctx,
    const Tensor& cond,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  return where_out(ctx, cond, a, b, out);
}

} // namespace native
} // namespace HiFi
} // namespace impl
