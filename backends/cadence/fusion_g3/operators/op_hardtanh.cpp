/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/cadence/fusion_g3/operators/operators.h>

#include <cmath>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/fusion_g3/operators/xt_macros.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::torch::executor::native::utils::extract_scalar;
using ::torch::executor::native::utils::get_scalar_dtype;

namespace impl {
namespace G3 {
namespace native {

Tensor& hardtanh_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Scalar& min,
    const Scalar& max,
    Tensor& out) {
  (void)ctx;

#ifdef OP_ARG_CHECK
  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      executorch::runtime::resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(in, out),
      InvalidArgument,
      out);
#endif

  ScalarType in_type = in.scalar_type();
  ScalarType min_type = get_scalar_dtype(min);
  ScalarType max_type = get_scalar_dtype(max);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, in_type == out_type, InvalidArgument, out);

  if (in_type == ScalarType::Float) {
    const float* const inp1_data = in.const_data_ptr<float>();
    float* const out_data = out.mutable_data_ptr<float>();
    float min_val, max_val;
    extract_scalar(min, &min_val);
    extract_scalar(max, &max_val);

    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_elm_clamp_scalar_f32_f32,
        out_data,
        inp1_data,
        min_val,
        max_val,
        out.numel());
  } else {
    ET_SWITCH_REALHBF16_TYPES(in_type, ctx, "hardtanh.out", CTYPE, [&]() {
      CTYPE min_casted;
      ET_SWITCH_SCALAR_OBJ_TYPES(
          min_type, ctx, "hardtanh.out", CTYPE_MIN, [&]() {
            CTYPE_MIN min_val;
            extract_scalar(min, &min_val);
            min_casted = static_cast<CTYPE>(min_val);
          });

      CTYPE max_casted;
      ET_SWITCH_SCALAR_OBJ_TYPES(
          max_type, ctx, "hardtanh.out", CTYPE_MAX, [&]() {
            CTYPE_MAX max_val;
            extract_scalar(max, &max_val);
            max_casted = static_cast<CTYPE>(max_val);
          });

      torch::executor::apply_unary_map_fn(
          [min_casted, max_casted](const CTYPE val_in) {
            return torch::executor::native::utils::min_override(
                torch::executor::native::utils::max_override(
                    val_in, min_casted),
                max_casted);
          },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    });
  }
  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
