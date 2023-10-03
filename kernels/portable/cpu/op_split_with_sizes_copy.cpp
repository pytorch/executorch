/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using TensorList = exec_aten::TensorList;

void split_with_sizes_copy_out(
    RuntimeContext& ctx,
    const Tensor& in,
    exec_aten::ArrayRef<int64_t> split_sizes,
    int64_t dim,
    TensorList out) {
  (void)ctx;
  // Support python-style negative indexing. Note that this op does not accept 0
  // dimensional input tensors.
  if (dim < 0) {
    dim += in.dim();
  }

  ET_KERNEL_CHECK(
      ctx,
      check_split_with_sizes_copy_args(in, split_sizes, dim, out),
      InvalidArgument,
      out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    expected_out_size[expected_out_dim++] = split_sizes[i];
    get_split_with_sizes_copy_out_target_size(
        in, split_sizes[i], dim, expected_out_size, &expected_out_dim);
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(out[i], {expected_out_size, expected_out_dim}) ==
            Error::Ok,
        InvalidArgument,
        out);
  }

  const size_t leading_dims = getLeadingDims(in, dim);
  const size_t trailing_dims = getTrailingDims(in, dim);
  const size_t step = in.size(dim) * trailing_dims;

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out[0].scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, __func__, CTYPE_IN, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, __func__, CTYPE_OUT, [&]() {
      const CTYPE_IN* in_data = in.const_data_ptr<CTYPE_IN>();
      for (size_t i = 0, e = out.size(); i < e; ++i) {
        size_t out_step = out[i].size(dim) * trailing_dims;
        if (out_step == 0) {
          continue;
        }
        const CTYPE_IN* src = in_data;
        CTYPE_OUT* dest = out[i].mutable_data_ptr<CTYPE_OUT>();
        for (size_t j = 0; j < leading_dims; ++j) {
          for (size_t k = 0; k < out_step; ++k) {
            dest[k] = convert<CTYPE_OUT, CTYPE_IN>(src[k]);
          }
          src += step;
          dest += out_step;
        }
        in_data += out_step;
      }
    });
  });
}

} // namespace native
} // namespace executor
} // namespace torch
