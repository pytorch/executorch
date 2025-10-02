/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::apply_unary_map_fn;
using torch::executor::Error;
using torch::executor::native::utils::extract_scalar;
using torch::executor::native::utils::get_scalar_dtype;
using torch::executor::native::utils::max_override;
using torch::executor::native::utils::min_override;

namespace impl {
namespace HiFi {
namespace native {

Tensor& hardtanh_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Scalar& min,
    const Scalar& max,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType min_type = get_scalar_dtype(min);
  ScalarType max_type = get_scalar_dtype(max);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, in_type == out_type, InvalidArgument, out);

  bool optimized = 1;
  if ((in_type != ScalarType::Float) || (out_type != ScalarType::Float))
    optimized = 0;

  if (optimized) {
    float* data_in = in.mutable_data_ptr<float>();
    float* data_out = out.mutable_data_ptr<float>();
    float min_val, max_val;
    extract_scalar(min, &min_val);
    extract_scalar(max, &max_val);
    xa_nn_vec_activation_min_max_f32_f32(
        data_out, data_in, min_val, max_val, in.numel());

    return out;
  }

  ET_SWITCH_REALHBF16_TYPES(in_type, ctx, "hardtanh.out", CTYPE, [&]() {
    CTYPE min_casted;
    ET_SWITCH_SCALAR_OBJ_TYPES(min_type, ctx, "hardtanh.out", CTYPE_MIN, [&]() {
      CTYPE_MIN min_val;
      extract_scalar(min, &min_val);
      min_casted = static_cast<CTYPE>(min_val);
    });

    CTYPE max_casted;
    ET_SWITCH_SCALAR_OBJ_TYPES(max_type, ctx, "hardtanh.out", CTYPE_MAX, [&]() {
      CTYPE_MAX max_val;
      extract_scalar(max, &max_val);
      max_casted = static_cast<CTYPE>(max_val);
    });

    apply_unary_map_fn(
        [min_casted, max_casted](const CTYPE val_in) {
          return min_override(max_override(val_in, min_casted), max_casted);
        },
        in.const_data_ptr<CTYPE>(),
        out.mutable_data_ptr<CTYPE>(),
        in.numel());
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
