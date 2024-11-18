/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& stack_out(
    KernelRuntimeContext& ctx,
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  (void)ctx;

  if (dim < 0) {
    dim += out.dim();
  }

  ET_KERNEL_CHECK(
      ctx, check_stack_args(tensors, dim, out), InvalidArgument, out);

  for (size_t i = 0; i < tensors.size(); ++i) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(tensors[i], out),
        InvalidArgument,
        out);
  }

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_stack_out_target_size(tensors, dim, expected_out_size, &expected_out_dim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  const size_t outer = getLeadingDims(out, dim);
  const size_t inner = getTrailingDims(out, dim);
  const size_t ninputs = tensors.size();

  const auto out_type = out.scalar_type();
  ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "stack.out", CTYPE_OUT, [&] {
    CTYPE_OUT* out_ptr = out.mutable_data_ptr<CTYPE_OUT>();
    for (size_t i = 0; i < outer; ++i) {
      for (size_t j = 0; j < ninputs; ++j) {
        const auto in_type = tensors[j].scalar_type();
        ET_SWITCH_REAL_TYPES_AND(
            Bool, in_type, ctx, "stack.out", CTYPE_IN, [&] {
              const CTYPE_IN* const in_ptr =
                  tensors[j].const_data_ptr<CTYPE_IN>() + i * inner;

              for (size_t k = 0; k < inner; ++k) {
                out_ptr[k] = static_cast<CTYPE_OUT>(in_ptr[k]);
              }
              out_ptr += inner;
            });
      }
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
