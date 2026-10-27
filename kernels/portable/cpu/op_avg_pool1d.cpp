/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <type_traits>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;

Tensor& avg_pool1d_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_avg_pool1d_args(in, kernel_size, stride, padding, ceil_mode, out),
      InvalidArgument,
      out);

  size_t output_ndim = 0;
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_avg_pool1d_out_target_size(
      in, kernel_size, stride, padding, ceil_mode, output_sizes, &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      output_size_is_valid({output_sizes, output_ndim}, 1),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "avg_pool1d.out";

  ET_SWITCH_FLOATHBF16_TYPES_AND(Long, in_type, ctx, op_name, CTYPE, [&]() {
    using ACC = std::conditional_t<
        std::is_same_v<CTYPE, executorch::aten::Half> ||
            std::is_same_v<CTYPE, executorch::aten::BFloat16>,
        float,
        CTYPE>;
    apply_kernel_1d_reduce_then_map_fn<CTYPE, ACC>(
        [](const CTYPE in_val, const ACC accum) {
          return accum + static_cast<ACC>(in_val);
        },
        [](const int64_t count, const ACC accum) {
          return static_cast<CTYPE>(accum / static_cast<ACC>(count));
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
