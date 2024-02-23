/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/distance_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace torch {
namespace executor {

bool check_pdist_args(const Tensor& in, double p, const Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(in, 2));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      p >= 0, "pdist only supports non-negative p values");
  return true;
}

void get_pdist_out_target_size(
    const Tensor& in,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = 1;
  size_t n = in.size(0);
  out_sizes[0] = n * (n - 1) / 2;
}

} // namespace executor
} // namespace torch
