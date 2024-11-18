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

Tensor& avg_pool2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    exec_aten::optional<int64_t> divisor_override,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_avg_pool2d_args(
          in,
          kernel_size,
          stride,
          padding,
          ceil_mode,
          count_include_pad,
          divisor_override,
          out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_avg_pool2d_out_target_size(
      in, kernel_size, stride, padding, ceil_mode, output_sizes, &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      output_size_is_valid({output_sizes, output_ndim}, 2),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();
  ET_SWITCH_FLOAT_TYPES_AND(Long, in_type, ctx, "avg_pool2d.out", CTYPE, [&]() {
    if (divisor_override.has_value()) {
      int64_t divisor = divisor_override.value();
      // If divisor_override is specified, then we don't need to use `count` in
      // the calculation. Simply sum x / divisor to get the output.
      apply_kernel_2d_reduce_then_map_fn<CTYPE>(
          [](const CTYPE in_val,
             int64_t in_idx,
             CTYPE accum,
             int64_t accum_idx) {
            // Average pooling does not track indexes, so return 0 for accum_idx
            return std::tuple<CTYPE, int64_t>(in_val + accum, 0);
          },
          [divisor](const int64_t count, const CTYPE accum) {
            return accum / static_cast<CTYPE>(divisor);
          },
          count_include_pad,
          in,
          kernel_size,
          stride,
          padding,
          {},
          out);
    } else {
      apply_kernel_2d_reduce_then_map_fn<CTYPE>(
          [](const CTYPE in_val,
             int64_t in_idx,
             CTYPE accum,
             int64_t accum_idx) {
            // Average pooling does not track indexes, so return 0 for accum_idx
            return std::tuple<CTYPE, int64_t>(in_val + accum, 0);
          },
          [](const int64_t count, const CTYPE accum) {
            return accum / static_cast<CTYPE>(count);
          },
          count_include_pad,
          in,
          kernel_size,
          stride,
          padding,
          {},
          out);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
