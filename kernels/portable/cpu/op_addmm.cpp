/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using Scalar = exec_aten::Scalar;

Tensor& addmm_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_addmm_args(in, mat1, mat2, beta, alpha, out),
      InvalidArgument,
      out);

  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_mm_out_target_size(mat1, mat2, output_sizes, &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensor_is_broadcastable_to(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, mat1, mat2, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  ScalarType alpha_dtype = utils::get_scalar_dtype(alpha);
  ScalarType beta_dtype = utils::get_scalar_dtype(beta);
  ET_SWITCH_REAL_TYPES_AND(
      Half, in.scalar_type(), ctx, "addmm.out", CTYPE, [&]() {
        ET_SWITCH_SCALAR_OBJ_TYPES(
            alpha_dtype, ctx, "addmm.out", ALPHA_T, [&]() {
              ET_SWITCH_SCALAR_OBJ_TYPES(
                  beta_dtype, ctx, "addmm.out", BETA_T, [&]() {
                    size_t m = mat1.size(0);
                    size_t n = mat1.size(1);
                    size_t p = mat2.size(1);

                    if (out.sizes() == in.sizes()) {
                      // vec_addmm assumes that no broadcasting is required.
                      vec_addmm<CTYPE, CTYPE>(
                          out.mutable_data_ptr<CTYPE>(),
                          in.const_data_ptr<CTYPE>(),
                          mat1.const_data_ptr<CTYPE>(),
                          mat2.const_data_ptr<CTYPE>(),
                          m,
                          n,
                          p,
                          convert<CTYPE>(beta.to<BETA_T>()),
                          convert<CTYPE>(alpha.to<ALPHA_T>()));
                    } else {
                      // If broadcasting is required, them compute the matmul
                      // and addition separately, using
                      // apply_binary_elementwise_fn to perform the addition
                      // while applying broadcasting
                      vec_matmul<CTYPE, CTYPE>(
                          out.mutable_data_ptr<CTYPE>(),
                          mat1.const_data_ptr<CTYPE>(),
                          mat2.const_data_ptr<CTYPE>(),
                          m,
                          n,
                          p);

                      CTYPE alpha_val = convert<CTYPE>(alpha.to<ALPHA_T>());
                      CTYPE beta_val = convert<CTYPE>(beta.to<BETA_T>());
                      apply_binary_elementwise_fn<CTYPE, CTYPE, CTYPE>(
                          [alpha_val, beta_val](
                              const CTYPE val_a, const CTYPE val_b) {
                            CTYPE a_casted = static_cast<CTYPE>(val_a);
                            CTYPE b_casted = static_cast<CTYPE>(val_b);
                            CTYPE value =
                                a_casted * alpha_val + b_casted * beta_val;

                            return value;
                          },
                          out,
                          in,
                          out);
                    }
                  });
            });
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
