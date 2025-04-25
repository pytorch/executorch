/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/padding_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace torch {
namespace executor {

bool check_padding_args(
    int64_t n,
    const Tensor& in,
    executorch::aten::ArrayRef<int64_t> padding,
    Tensor& out,
    bool reflection) {
  ET_LOG_AND_RETURN_IF_FALSE(static_cast<int64_t>(padding.size()) == 2 * n);
  ET_LOG_AND_RETURN_IF_FALSE(in.dim() == n + 1 || in.dim() == n + 2);
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  for (const auto i : c10::irange(1, n + 1)) {
    ET_LOG_AND_RETURN_IF_FALSE(
        in.size(in.dim() - i) + padding[2 * i - 2] + padding[2 * i - 1] >= 0);
    if (reflection) {
      ET_LOG_AND_RETURN_IF_FALSE(
          padding[2 * i - 2] < in.size(in.dim() - i) &&
          padding[2 * i - 1] < in.size(in.dim() - i));
    }
  }
  return true;
}

void get_padding_out_target_size(
    int64_t n,
    const Tensor& in,
    executorch::aten::ArrayRef<int64_t> padding,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();
  for (const auto i : c10::irange(in.dim())) {
    out_sizes[i] = in.size(i);
  }
  for (const auto i : c10::irange(1, n + 1)) {
    out_sizes[in.dim() - i] =
        in.size(in.dim() - i) + padding[2 * i - 2] + padding[2 * i - 1];
  }
}

} // namespace executor
} // namespace torch
