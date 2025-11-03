/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/common/xt_macros.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;

namespace impl {
namespace G3 {
namespace native {

Tensor& where_self_out(
    KernelRuntimeContext& ctx,
    const Tensor& cond,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
#ifdef OP_ARG_CHECK
  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(cond, a, b, out),
      InvalidArgument,
      out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_to_broadcast_target_size(a, b, cond, out) ==
          Error::Ok,
      InvalidArgument,
      out);
#endif

  static constexpr const char op_name[] = "where.self_out";

  int kTensorDimensionLimit = 5;

  int cond_shape[kTensorDimensionLimit];
  int inp1_shape[kTensorDimensionLimit];
  int inp2_shape[kTensorDimensionLimit];
  int out_shape[kTensorDimensionLimit];

  bool broadcast = false;

  int max1_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  int max2_dim = cond.dim() > out.dim() ? cond.dim() : out.dim();
  int max_dim = max1_dim > max2_dim ? max1_dim : max2_dim;

  bool optimized = true;

  for (int i = 0; i < max_dim; i++) {
    out_shape[i] = 1;
    cond_shape[i] = 1;
    inp1_shape[i] = 1;
    inp2_shape[i] = 1;
  }

  int offset_out = max_dim - out.dim();
  int offset_cond = max_dim - cond.dim();
  int offset_inp1 = max_dim - a.dim();
  int offset_inp2 = max_dim - b.dim();

  for (int i = 0; i < out.dim(); i++) {
    out_shape[i + offset_out] = out.size(i);
  }
  for (int i = 0; i < cond.dim(); i++) {
    cond_shape[i + offset_cond] = cond.size(i);
  }
  for (int i = 0; i < a.dim(); i++) {
    inp1_shape[i + offset_inp1] = a.size(i);
  }
  for (int i = 0; i < b.dim(); i++) {
    inp2_shape[i + offset_inp2] = b.size(i);
  }

  /*find broadcast*/
  for (int i = 0; i < max_dim; i++) {
    if (((cond_shape[i]) != (out_shape[i])) ||
        ((inp1_shape[i]) != (out_shape[i])) ||
        ((inp2_shape[i]) != (out_shape[i]))) {
      broadcast = true;
    }
  }

  if (((broadcast) && (max_dim > kTensorDimensionLimit)) ||
      (!((a.scalar_type() == ScalarType::Float) &&
         (b.scalar_type() == ScalarType::Float) &&
         (cond.scalar_type() == ScalarType::Bool) &&
         (out.scalar_type() == ScalarType::Float)))) {
    optimized = false;
  }

  if (optimized) {
    const unsigned char* const cond_data = cond.const_data_ptr<unsigned char>();
    const float* const inp1_data = a.const_data_ptr<float>();
    const float* const inp2_data = b.const_data_ptr<float>();
    float* const out_data = out.mutable_data_ptr<float>();

    if (broadcast) {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_where_broadcast_5D_f32xf32_f32,
          out_data,
          out_shape,
          inp1_data,
          inp1_shape,
          inp2_data,
          inp2_shape,
          cond_data,
          cond_shape,
          max_dim);
    } else {
      XT_KERNEL_CHECK(
          ctx,
          out,
          xa_nn_elm_where_f32xf32_f32,
          out_data,
          inp1_data,
          inp2_data,
          cond_data,
          out.numel());
    }
  } else {
    // Common Dtype
    ScalarType common_type =
        executorch::runtime::promoteTypes(a.scalar_type(), b.scalar_type());

    // Check Common Dtype
    ET_KERNEL_CHECK(
        ctx, common_type == out.scalar_type(), InvalidArgument, out);

    // Compute Dtype
    ScalarType compute_type =
        torch::executor::native::utils::get_compute_type(common_type);

    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      torch::executor::native::utils::apply_tritensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name>(
          [](const CTYPE_COMPUTE val_a,
             const CTYPE_COMPUTE val_b,
             const CTYPE_COMPUTE val_c) { return val_c ? val_a : val_b; },
          ctx,
          a,
          torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
          b,
          torch::executor::native::utils::SupportedTensorDtypes::REALHBBF16,
          cond,
          torch::executor::native::utils::SupportedTensorDtypes::BOOL_OR_BYTE,
          out,
          torch::executor::native::utils::SupportedTensorDtypes::
              SAME_AS_COMMON);
    });
  }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
