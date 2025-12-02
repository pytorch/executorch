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

bool check_grid_sampler_2d_args(
    const Tensor& input,
    const Tensor& grid,
    const Tensor& out) {
  // Input must be 4D (N, C, H, W)
  ET_LOG_AND_RETURN_IF_FALSE(input.dim() == 4);
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(input));

  // Grid must be 4D (N, H_out, W_out, 2)
  ET_LOG_AND_RETURN_IF_FALSE(grid.dim() == 4);
  ET_LOG_AND_RETURN_IF_FALSE(grid.size(3) == 2);

  // Output must be 4D (N, C, H_out, W_out)
  ET_LOG_AND_RETURN_IF_FALSE(out.dim() == 4);

  // Batch sizes must match
  ET_LOG_AND_RETURN_IF_FALSE(input.size(0) == grid.size(0));
  ET_LOG_AND_RETURN_IF_FALSE(input.size(0) == out.size(0));

  // Channel dimension must match between input and output
  ET_LOG_AND_RETURN_IF_FALSE(input.size(1) == out.size(1));

  // Output spatial dimensions must match grid dimensions
  ET_LOG_AND_RETURN_IF_FALSE(out.size(2) == grid.size(1));
  ET_LOG_AND_RETURN_IF_FALSE(out.size(3) == grid.size(2));

  // Input and output must have same dtype
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(input, out));

  // Grid and input must have same dtype
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(input, grid));

  // Output must have same dim order as input
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dim_order(input, out));

  return true;
}

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

  // Resize output tensor to [N, C, H_out, W_out]
  std::array<exec_aten::SizesType, 4> out_sizes = {
      static_cast<exec_aten::SizesType>(input.size(0)),
      static_cast<exec_aten::SizesType>(input.size(1)),
      static_cast<exec_aten::SizesType>(grid.size(1)),
      static_cast<exec_aten::SizesType>(grid.size(2))};

  Error err = resize_tensor(out, {out_sizes.data(), 4});
  ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok,
      InvalidArgument,
      "Failed to resize output tensor");

  return Error::Ok;
}

} // namespace executor
} // namespace torch