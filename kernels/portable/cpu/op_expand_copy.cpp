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

size_t calculate_output_sizes(
    exec_aten::ArrayRef<SizesType> self_sizes,
    exec_aten::ArrayRef<int64_t> expand_sizes,
    SizesType* output_sizes) {
  auto j{expand_sizes.size()};
  for (size_t i{self_sizes.size()}; i > 0 && j > 0;) {
    --i;
    --j;

    output_sizes[j] = expand_sizes[j];

    if (expand_sizes[j] == -1) {
      // -1 can use for replacing any corresponding dimension
      output_sizes[j] = self_sizes[i];
    } else if (self_sizes[i] != 1) {
      ET_CHECK_MSG(
          expand_sizes[j] == self_sizes[i],
          "The expanded size of the tensor (%zu) must match the existing size (%zu) at non-singleton dimension %zu.",
          (size_t)expand_sizes[j],
          (size_t)self_sizes[i],
          i);
    }
  }

  // The leading expand_sizes cannot be negative
  while (j > 0) {
    --j;
    output_sizes[j] = expand_sizes[j];
    ET_CHECK_MSG(
        expand_sizes[j] >= 0,
        "The expanded size of the tensor (%zu) isn't allowed in a leading, non-existing dimension %zu",
        (size_t)expand_sizes[j],
        j);
  }

  return expand_sizes.size();
}

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

void check_output_tensor(
    const Tensor& self,
    exec_aten::ArrayRef<exec_aten::SizesType> output_sizes,
    const Tensor& out) {
  ET_CHECK_SAME_DTYPE2(self, out);

  const auto& out_sizes = out.sizes();

  ET_CHECK_MSG(
      output_sizes.size() == out_sizes.size(),
      "Number of output tensor sizes (%zu) must match the number of expanded output sizes (%zu)",
      output_sizes.size(),
      out_sizes.size());

  // Make sure the output shape is correct
  for (size_t i{0}; i < output_sizes.size(); i++) {
    ET_CHECK_MSG(
        output_sizes[i] == out_sizes[i],
        "Size of output tensor (%d) must match size of expanded output (%d) at dimension (%zu)",
        out_sizes[i],
        output_sizes[i],
        i);
  }
}
} // namespace

Tensor& expand_copy_out(
    RuntimeContext& ctx,
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
  const auto output_sizes_size{
      calculate_output_sizes(self_sizes, expand_sizes, output_sizes)};

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_sizes_size}) == Error::Ok,
      InvalidArgument,
      out);

  // Check that the output tensor is the same shape as the mapped expand
  check_output_tensor(self, {output_sizes, output_sizes_size}, out);

  // Holds the result of expand_sizes converted to repeat sizes
  int64_t repeats[kTensorDimensionLimit];
  const auto repeats_size{map_expand_to_repeats(
      self_sizes, expand_sizes, repeats, kTensorDimensionLimit)};

  repeat_tensor(self, {repeats, repeats_size}, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
