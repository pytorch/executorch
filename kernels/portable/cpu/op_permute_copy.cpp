/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using SizesType = exec_aten::SizesType;
using Tensor = exec_aten::Tensor;
using IntArrayRef = exec_aten::ArrayRef<int64_t>;

namespace {

void increment_coordinate_permuted(
    const Tensor& tensor,
    size_t* const coordinate,
    IntArrayRef dims) {
  for (int i = dims.size() - 1; i >= 0; i--) {
    size_t d = dims[i] >= 0 ? dims[i] : dims[i] + tensor.dim();
    coordinate[d]++;
    if (coordinate[d] == tensor.size(d)) {
      coordinate[d] = 0;
    } else {
      return;
    }
  }
}

} // namespace

Tensor& permute_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dims,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_permute_copy_args(in, dims, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_permute_copy_out_target_size(
      in, dims, expected_out_size, &expected_out_dim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  const auto in_type = out.scalar_type();

  size_t in_coord[kTensorDimensionLimit] = {0};
  size_t trailing_dims_memo[kTensorDimensionLimit];
  executorch::runtime::memoizeTrailingDims(in, trailing_dims_memo);

  // in and out must be the same dtype
  ET_SWITCH_ALL_TYPES(in_type, ctx, "permute_copy.out", CTYPE, [&] {
    const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

    for (size_t i = 0; i < out.numel(); ++i) {
      out_data[i] =
          in_data[executorch::runtime::coordinateToIndexWithTrailingDimsMemo(
              in, in_coord, trailing_dims_memo)];
      increment_coordinate_permuted(in, in_coord, dims);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
