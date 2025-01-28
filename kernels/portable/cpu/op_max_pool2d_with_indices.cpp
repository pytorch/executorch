/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using IntArrayRef = exec_aten::ArrayRef<int64_t>;

std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& out,
    Tensor& indices) {
  std::tuple<Tensor&, Tensor&> ret_val(out, indices);

  ET_KERNEL_CHECK(
      ctx,
      check_max_pool2d_with_indices_args(
          in, kernel_size, stride, padding, dilation, ceil_mode, out, indices),
      InvalidArgument,
      ret_val);

  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_max_pool2d_with_indices_out_target_size(
      in,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output_sizes,
      &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      output_size_is_valid({output_sizes, output_ndim}, 2),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(indices, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ScalarType in_type = in.scalar_type();
  ET_SWITCH_REAL_TYPES(
      in_type, ctx, "max_pool2d_with_indices.out", CTYPE, [&]() {
        apply_kernel_2d_reduce_then_map_fn<CTYPE>(
            [](const CTYPE in_val,
               const int64_t in_idx,
               const CTYPE accum,
               const int64_t accum_idx) {
              if (in_val > accum) {
                return std::tuple<CTYPE, int64_t>(in_val, in_idx);
              }
              return std::tuple<CTYPE, int64_t>(accum, accum_idx);
            },
            // Max pooling does not need to post-process the accumulated output
            [](const int64_t count, const CTYPE accum) { return accum; },
            /*include_pad=*/false,
            in,
            kernel_size,
            stride,
            padding,
            dilation,
            out,
            {indices});
      });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
