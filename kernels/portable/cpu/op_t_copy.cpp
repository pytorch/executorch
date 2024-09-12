/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/transpose_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using SizesType = exec_aten::SizesType;
using StridesType = exec_aten::StridesType;
using Tensor = exec_aten::Tensor;

/**
 * Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
 * 0-D and 1-D tensors are returned as is. When input is a 2-D tensor this
 * is equivalent to transpose(input, 0, 1).
 * t_copy.out(Tensor self, Tensor(a!) out)
 */
Tensor& t_copy_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(ctx, check_t_copy_args(in, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();

  if (in.dim() < 2) {
    // Resize for dynamic shape
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

    if (in.numel() > 0) {
      ET_SWITCH_ALL_TYPES(in_type, ctx, __func__, CTYPE, [&]() {
        const CTYPE* in_data = in.const_data_ptr<CTYPE>();
        CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
        memcpy(out_data, in_data, in.nbytes());
      });
    }

    return out;
  }

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_transpose_out_target_size(in, 1, 0, expected_out_size, &expected_out_dim);

  // Resize for dynamic shape
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_SWITCH_ALL_TYPES(in_type, ctx, __func__, CTYPE, [&] {
    transpose_tensors<CTYPE>(in, 1, 0, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
