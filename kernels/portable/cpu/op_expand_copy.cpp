/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/kernels/portable/cpu/util/repeat_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <sys/types.h>

#include <cstring>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using Scalar = exec_aten::Scalar;
using SizesType = exec_aten::SizesType;

constexpr const size_t kTensorDimensionLimit{16};

namespace {

size_t map_expand_to_repeats(
    exec_aten::ArrayRef<SizesType> self_sizes,
    exec_aten::ArrayRef<int64_t> expand_sizes,
    int64_t* repeats,
    const size_t repeats_size) {
  auto j{expand_sizes.size()};
  for (size_t i{self_sizes.size()}; i > 0 && j > 0;) {
    --i;
    --j;

    // Default, just copy the expand size to repeat
    repeats[j] = expand_sizes[j];
    if (expand_sizes[j] == -1 || expand_sizes[j] == self_sizes[i]) {
      repeats[j] = 1;
    }
  }

  while (j > 0) {
    --j;
    repeats[j] = expand_sizes[j];
  }

  return expand_sizes.size();
}
} // namespace

Tensor& expand_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    ArrayRef<int64_t> expand_sizes,
    bool implicit,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_expand_copy_args(self, expand_sizes, implicit, out),
      InvalidArgument,
      out);

  const auto& self_sizes = self.sizes();

  // Holds the result of converting -1 to the original dim sizes
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  size_t output_rank = 0;
  ET_KERNEL_CHECK(
      ctx,
      get_expand_copy_out_target_size(
          self_sizes, expand_sizes, output_sizes, &output_rank),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_rank}) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(self), InvalidArgument, out);

  // Holds the result of expand_sizes converted to repeat sizes
  int64_t repeats[kTensorDimensionLimit];
  const auto repeats_size{map_expand_to_repeats(
      self_sizes, expand_sizes, repeats, kTensorDimensionLimit)};

  ET_KERNEL_CHECK(
      ctx,
      repeat_tensor(self, {repeats, repeats_size}, out) == Error::Ok,
      InvalidArgument,
      out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
