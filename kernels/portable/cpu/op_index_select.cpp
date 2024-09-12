/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>
#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& index_select_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx, check_index_select_args(in, dim, index, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  size_t expected_ndim = 0;
  Tensor::SizesType expected_size[kTensorDimensionLimit];
  get_index_select_out_target_size(
      in, dim, index, expected_size, &expected_ndim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_size, expected_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  if (in.dim() == 0) {
    memcpy(out.mutable_data_ptr(), in.const_data_ptr(), in.nbytes());
    return out;
  }

  size_t leading_dims = getLeadingDims(in, dim);
  size_t trailing_dims = getTrailingDims(in, dim);

  if (leading_dims == 0 || trailing_dims == 0) {
    return out;
  }

  size_t out_dim_length = out.size(dim);
  size_t in_dim_length = in.size(dim);

  size_t length_per_step = trailing_dims * in.element_size();

  const char* input_data = in.const_data_ptr<char>();
  char* out_data = out.mutable_data_ptr<char>();

  ScalarType ix_type = index.scalar_type();

  ET_SWITCH_TWO_TYPES(
      Long, Int, ix_type, ctx, "index_select.out", CTYPE, [&]() {
        const CTYPE* const index_arr = index.mutable_data_ptr<CTYPE>();
        for (int i = 0; i < leading_dims; i++) {
          const char* src = input_data + i * in_dim_length * length_per_step;
          char* dest = out_data + i * out_dim_length * length_per_step;
          for (auto j = 0; j < out_dim_length; j++) {
            const char* copy_src = src + index_arr[j] * length_per_step;
            memcpy(dest, copy_src, length_per_step);
            dest += length_per_step;
          }
        }
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
