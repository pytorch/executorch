/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& rsub_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_realhb_type(out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType alpha_type = utils::get_scalar_dtype(alpha);
  ScalarType common_type = utils::promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, check_alpha_type(alpha_type, common_type), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, tensor_is_real_type(out), InvalidArgument, out);

  ET_SWITCH_REAL_TYPES(a_type, ctx, "rsub.Scalar_out", CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_REAL_TYPES(
        b_type, ctx, "rsub.Scalar_out", CTYPE_B, [&]() {
          ET_SWITCH_REAL_TYPES(
              common_type, ctx, "rsub.Scalar_out", CTYPE_IN, [&]() {
                ET_SWITCH_REAL_TYPES(
                    out_type, ctx, "rsub.Scalar_out", CTYPE_OUT, [&]() {
                      CTYPE_B b_val;
                      utils::extract_scalar(b, &b_val);
                      CTYPE_IN b_casted = static_cast<CTYPE_IN>(b_val);
                      CTYPE_IN alpha_val;
                      utils::extract_scalar(alpha, &alpha_val);

                      apply_unary_map_fn(
                          [b_casted, alpha_val](const CTYPE_A val_a) {
                            CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                            CTYPE_IN value = b_casted - alpha_val * a_casted;
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

} // namespace native
} // namespace executor
} // namespace torch
