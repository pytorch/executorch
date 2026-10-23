/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tuple>
#include <type_traits>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;

Tensor& avg_pool3d_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_avg_pool3d_args(
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
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_avg_pool3d_out_target_size(
      in, kernel_size, stride, padding, ceil_mode, output_sizes, &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      output_size_is_valid({output_sizes, output_ndim}, 3),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "avg_pool3d.out";

  ET_SWITCH_FLOATHBF16_TYPES_AND(Long, in_type, ctx, op_name, CTYPE, [&]() {
    using AccType = std::conditional_t<
        std::is_same_v<CTYPE, executorch::aten::Half> ||
            std::is_same_v<CTYPE, executorch::aten::BFloat16>,
        float,
        CTYPE>;
    apply_kernel_3d_reduce_then_map_fn<CTYPE, AccType>(
        [](const CTYPE in_val,
           int64_t in_idx,
           AccType accum,
           int64_t accum_idx) {
          return std::tuple<AccType, int64_t>(
              static_cast<AccType>(in_val) + accum, 0);
        },
        [divisor_override](const int64_t count, const AccType accum) {
          return accum / static_cast<AccType>(divisor_override.value_or(count));
        },
        count_include_pad,
        in,
        kernel_size,
        stride,
        padding,
        out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
