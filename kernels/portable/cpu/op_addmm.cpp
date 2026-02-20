/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using Scalar = executorch::aten::Scalar;

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
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
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

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "addmm.out";

  ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&]() {
    CTYPE alpha_val = utils::scalar_to<CTYPE>(alpha);
    CTYPE beta_val = utils::scalar_to<CTYPE>(beta);
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
          beta_val,
          alpha_val);
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

      utils::apply_bitensor_elementwise_fn<
          CTYPE,
          op_name,
          utils::SupportedTensorDtypes::REALHBF16>(
          [alpha_val, beta_val](const auto& val_a, const auto& val_b) {
            return val_a * alpha_val + val_b * beta_val;
          },
          ctx,
          out,
          utils::SupportedTensorDtypes::REALHBF16,
          in,
          utils::SupportedTensorDtypes::REALHBF16,
          out);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
