/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/extension/llm/custom_ops/spinquant/fast_hadamard_transform.h>
#include <executorch/kernels/optimized/utils/llvmMathExtras.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h> // For apply_over_dim.
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& fast_hadamard_transform_out(
    RuntimeContext& ctx,
    const Tensor& mat,
    Tensor& out) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, mat.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, mat.scalar_type() == out.scalar_type(), InvalidArgument, out);

  if (mat.dim() == 0 || mat.numel() == 0) {
    return out;
  }

  ET_KERNEL_CHECK(
      ctx,
      is_contiguous_dim_order(mat.dim_order().data(), mat.dim()),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      is_contiguous_dim_order(out.dim_order().data(), out.dim()),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      mat.strides().back() == 1,
      InvalidArgument,
      out,
      "input matrix that isn't contiguous in the last dimension is not supported!");

  const auto last_dim_size = mat.sizes().back();
  const auto divisible_by_28 = last_dim_size % 28 == 0;
  auto power_of_two_size = divisible_by_28 ? last_dim_size / 28 : last_dim_size;
  ET_KERNEL_CHECK_MSG(
      ctx,
      (power_of_two_size & (power_of_two_size - 1)) == 0,
      InvalidArgument,
      out,
      "This implementation requires power-of-2 (or power-of-2 * 28) input size in the last dimension!");

  const auto log2_power_of_two_size = executorch::llvm::countTrailingZeros(
      static_cast<unsigned int>(power_of_two_size),
      executorch::llvm::ZeroBehavior::ZB_Undefined);

  ET_SWITCH_FLOATH_TYPES(mat.scalar_type(), ctx, __func__, CTYPE, [&] {
    const CTYPE* const mat_data = mat.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

    std::memcpy(out_data, mat_data, mat.numel() * sizeof(CTYPE));

    if (divisible_by_28) {
      apply_over_dim(
          [log2_power_of_two_size, out_data](
              const size_t size, const size_t stride, const size_t base) {
            executorch::fast_hadamard_transform_28N(
                out_data + base, log2_power_of_two_size);
          },
          out,
          out.dim() - 1);
    } else {
      apply_over_dim(
          [log2_power_of_two_size, out_data](
              const size_t size, const size_t stride, const size_t base) {
            executorch::fast_hadamard_transform(
                out_data + base, log2_power_of_two_size);
          },
          out,
          out.dim() - 1);
    }
  });
  return out;
}
} // namespace native
} // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(
    llama,
    "fast_hadamard_transform.out",
    torch::executor::native::fast_hadamard_transform_out);
