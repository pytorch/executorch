/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/grid_sampler_2d_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace torch {
namespace executor {

Error check_grid_sampler_2d_args_and_resize_out(
    const Tensor& input,
    const Tensor& grid,
    Tensor& out) {
  // Input must be 4D (N, C, H, W)
  ET_CHECK_OR_RETURN_ERROR(
      input.dim() == 4,
      InvalidArgument,
      "Input must be 4D, got %zu dimensions",
      static_cast<size_t>(input.dim()));

  ET_CHECK_OR_RETURN_ERROR(
      tensor_is_default_dim_order(input),
      InvalidArgument,
      "Input must be in NCHW format");

  // Grid must be 4D (N, H_out, W_out, 2)
  ET_CHECK_OR_RETURN_ERROR(
      grid.dim() == 4,
      InvalidArgument,
      "Grid must be 4D, got %zu dimensions",
      static_cast<size_t>(grid.dim()));

  ET_CHECK_OR_RETURN_ERROR(
      grid.size(3) == 2,
      InvalidArgument,
      "Grid last dimension must be 2, got %ld",
      static_cast<long>(grid.size(3)));

  // Batch sizes must match
  ET_CHECK_OR_RETURN_ERROR(
      input.size(0) == grid.size(0),
      InvalidArgument,
      "Input and grid batch sizes must match, got input=%ld, grid=%ld",
      static_cast<long>(input.size(0)),
      static_cast<long>(grid.size(0)));

  // Input and grid must have same dtype
  ET_CHECK_OR_RETURN_ERROR(
      tensors_have_same_dtype(input, grid),
      InvalidArgument,
      "Input and grid must have same dtype");

  // Input and output must have the same dtype
  ET_CHECK_OR_RETURN_ERROR(
      tensors_have_same_dtype(input, out),
      InvalidArgument,
      "Input and output must have the same dtype");

  // Resize output tensor to [N, C, H_out, W_out]
  std::array<exec_aten::SizesType, 4> out_sizes = {
      static_cast<exec_aten::SizesType>(input.size(0)),
      static_cast<exec_aten::SizesType>(input.size(1)),
      static_cast<exec_aten::SizesType>(grid.size(1)),
      static_cast<exec_aten::SizesType>(grid.size(2))};

  Error err = resize_tensor(out, {out_sizes.data(), 4});
  ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok, InvalidArgument, "Failed to resize output tensor");

  return Error::Ok;
}

} // namespace executor
} // namespace torch