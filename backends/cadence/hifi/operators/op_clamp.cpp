/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdbool.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::RuntimeContext;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::canCast;
using executorch::runtime::isFloatingType;
using executorch::runtime::isIntegralType;
using executorch::runtime::promoteTypes;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::apply_ternary_elementwise_fn;
using torch::executor::Error;
using torch::executor::resize_to_broadcast_target_size;
using torch::executor::native::utils::apply_tritensor_elementwise_fn;
using torch::executor::native::utils::apply_unitensor_elementwise_fn;
using torch::executor::native::utils::extract_scalar;
using torch::executor::native::utils::get_compute_type;
using torch::executor::native::utils::get_scalar_dtype;
using torch::executor::native::utils::max_override;
using torch::executor::native::utils::min_override;
using torch::executor::native::utils::promote_type_with_scalar;
using torch::executor::native::utils::scalar_to;
using torch::executor::native::utils::SupportedTensorDtypes;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

Tensor& clamp_Tensor_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const std::optional<Tensor>& min_opt,
    const std::optional<Tensor>& max_opt,
    Tensor& out) {
  (void)ctx;

  bool has_min = min_opt.has_value();
  bool has_max = max_opt.has_value();

  ET_KERNEL_CHECK_MSG(
      ctx,
      has_min || has_max,
      InvalidArgument,
      out,
      "At least one of 'min' or 'max' must not be None");

  const Tensor& min = has_min ? min_opt.value() : in;
  const Tensor& max = has_max ? max_opt.value() : in;

  // Common Dtype
  ScalarType common_type = in.scalar_type();
  if (has_min) {
    common_type = promoteTypes(common_type, min.scalar_type());
  }
  if (has_max) {
    common_type = promoteTypes(common_type, max.scalar_type());
  }

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx, canCast(common_type, out.scalar_type()), InvalidArgument, out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, min, max, out),
      InvalidArgument,
      out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(in, min, max, out) == Error::Ok,
      InvalidArgument,
      out);

  // Compute Dtype
  ScalarType compute_type = get_compute_type(common_type);

  constexpr int kNnlibMaxDim =
      4; /*fallback to not optimised if broadcast and dim > 4 */

  ScalarType in_type = in.scalar_type();
  ScalarType min_type = min.scalar_type();
  ScalarType max_type = max.scalar_type();

  bool in_is_broadcasted = !out.sizes().equals(in.sizes());
  bool min_is_broadcasted = !out.sizes().equals(min.sizes());
  bool max_is_broadcasted = !out.sizes().equals(max.sizes());
  bool broadcast =
      (in_is_broadcasted || min_is_broadcasted || max_is_broadcasted);

  int max_dim = in.dim() > min.dim() ? in.dim() : min.dim();
  max_dim = max.dim() > max_dim ? max.dim() : max_dim;
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  bool optimized = true;
  bool fall_back = false;
  if ((in_type != ScalarType::Float) || (min_type != ScalarType::Float) ||
      (max_type != ScalarType::Float))
    optimized = false;
  if ((broadcast == true) && (max_dim > kNnlibMaxDim))
    optimized = false;

  if (optimized) {
    if (!has_min) {
      const float* const max_data = max.const_data_ptr<float>();
      const float* const inp_data = in.const_data_ptr<float>();
      float* const out_data = out.mutable_data_ptr<float>();
      if (broadcast) {
        int out_shape[kNnlibMaxDim];
        int inp_shape[kNnlibMaxDim];
        int max_shape[kNnlibMaxDim];

        for (int i = 0; i < kNnlibMaxDim; i++) {
          out_shape[i] = 1;
          inp_shape[i] = 1;
          max_shape[i] = 1;
        }

        int max_dim = max.dim(), inp_dim = in.dim(), out_dim = out.dim();
        int off_o = kNnlibMaxDim - out_dim;
        int off_max = kNnlibMaxDim - max_dim;
        int off_inp = kNnlibMaxDim - inp_dim;
        for (int i = 0; i < out_dim; i++) {
          out_shape[i + off_o] = out.size(i);
        }
        for (int i = 0; i < max_dim; i++) {
          max_shape[i + off_max] = max.size(i);
        }
        for (int i = 0; i < inp_dim; i++) {
          inp_shape[i + off_inp] = in.size(i);
        }

        WORD32 ret_val = xa_nn_elm_minimum_broadcast_4D_f32xf32_f32(
            out_data, out_shape, inp_data, inp_shape, max_data, max_shape);

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

      } else {
        WORD32 ret_val = xa_nn_elm_minimum_f32xf32_f32(
            out_data, inp_data, max_data, out.numel());

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
      }
    } else if (!has_max) {
      const float* const min_data = min.const_data_ptr<float>();
      const float* const inp_data = in.const_data_ptr<float>();
      float* const out_data = out.mutable_data_ptr<float>();
      if (broadcast == 1) {
        int out_shape[kNnlibMaxDim];
        int inp_shape[kNnlibMaxDim];
        int min_shape[kNnlibMaxDim];

        for (int i = 0; i < kNnlibMaxDim; i++) {
          out_shape[i] = 1;
          inp_shape[i] = 1;
          min_shape[i] = 1;
        }

        int min_dim = min.dim(), max_dim = max.dim(), inp_dim = in.dim(),
            out_dim = out.dim();
        int off_o = kNnlibMaxDim - out_dim;
        int off_min = kNnlibMaxDim - min_dim;
        int off_inp = kNnlibMaxDim - inp_dim;
        for (int i = 0; i < out_dim; i++)
          out_shape[i + off_o] = out.size(i);
        for (int i = 0; i < min_dim; i++)
          min_shape[i + off_min] = min.size(i);
        for (int i = 0; i < inp_dim; i++)
          inp_shape[i + off_inp] = in.size(i);
        WORD32 ret_val = xa_nn_elm_maximum_broadcast_4D_f32xf32_f32(
            out_data, out_shape, inp_data, inp_shape, min_data, min_shape);

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

      } else {
        WORD32 ret_val = xa_nn_elm_maximum_f32xf32_f32(
            out_data, inp_data, min_data, out.numel());

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
      }
    } else {
      const float* const min_data = min.const_data_ptr<float>();
      const float* const max_data = max.const_data_ptr<float>();
      const float* const inp_data = in.const_data_ptr<float>();
      float* const out_data = out.mutable_data_ptr<float>();
      if (broadcast == 1) {
        int out_shape[kNnlibMaxDim];
        int inp_shape[kNnlibMaxDim];
        int min_shape[kNnlibMaxDim];
        int max_shape[kNnlibMaxDim];

        for (int i = 0; i < kNnlibMaxDim; i++) {
          out_shape[i] = 1;
          inp_shape[i] = 1;
          min_shape[i] = 1;
          max_shape[i] = 1;
        }

        int min_dim = min.dim(), max_dim = max.dim(), inp_dim = in.dim(),
            out_dim = out.dim();
        int off_o = kNnlibMaxDim - out_dim;
        int off_min = kNnlibMaxDim - min_dim;
        int off_max = kNnlibMaxDim - max_dim;
        int off_inp = kNnlibMaxDim - inp_dim;
        for (int i = 0; i < out_dim; i++)
          out_shape[i + off_o] = out.size(i);
        for (int i = 0; i < min_dim; i++)
          min_shape[i + off_min] = min.size(i);

        for (int i = 0; i < max_dim; i++)
          max_shape[i + off_max] = max.size(i);

        for (int i = 0; i < inp_dim; i++)
          inp_shape[i + off_inp] = in.size(i);

        if (inp_shape[0] != out_shape[0] || inp_shape[1] != out_shape[1] ||
            inp_shape[2] != out_shape[2] || inp_shape[3] != out_shape[3]) {
          void* p_scratch = (void*)kernels::allocate_temp_memory(
              ctx,
              (out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]) *
                  sizeof(int));

          ET_KERNEL_CHECK(
              ctx, p_scratch != nullptr, MemoryAllocationFailed, out);

          const FLOAT32* p_brd_cond = (const FLOAT32*)p_scratch;
          xa_nn_broadcast_32_32(
              (WORD32*)p_brd_cond, out_shape, (WORD32*)inp_data, inp_shape, 4);

          for (int i = 0; i < 4; i++) {
            inp_shape[i] = out_shape[i];
          }

          WORD32 ret_val = xa_nn_elm_clamp_broadcast_4D_f32Xf32xf32_f32(
              out_data,
              out_shape,
              p_brd_cond,
              inp_shape,
              min_data,
              min_shape,
              max_data,
              max_shape);

          ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

        } else {
          WORD32 ret_val = xa_nn_elm_clamp_broadcast_4D_f32Xf32xf32_f32(
              out_data,
              out_shape,
              inp_data,
              inp_shape,
              min_data,
              min_shape,
              max_data,
              max_shape);

          ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
        }
      } else {
        WORD32 ret_val = xa_nn_elm_clamp_f32xf32xf32_f32(
            out_data, inp_data, min_data, max_data, out.numel());

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
      }
    }
    return out;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "clamp.Tensor_out";

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    apply_tritensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [has_min, has_max](
            const CTYPE_COMPUTE val_in,
            const CTYPE_COMPUTE val_min,
            const CTYPE_COMPUTE val_max) {
          CTYPE_COMPUTE val_out = val_in;
          if (has_min) {
            val_out = max_override(val_out, val_min);
          }
          if (has_max) {
            val_out = min_override(val_out, val_max);
          }
          return val_out;
        },
        ctx,
        in,
        SupportedTensorDtypes::REALHBBF16,
        min,
        SupportedTensorDtypes::REALHBBF16,
        max,
        SupportedTensorDtypes::REALHBBF16,
        out,
        SupportedTensorDtypes::REALHBBF16);
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
