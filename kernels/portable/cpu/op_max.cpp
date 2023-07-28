/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using ScalarType = exec_aten::ScalarType;
using SizesType = exec_aten::SizesType;
using Tensor = exec_aten::Tensor;

namespace {

void check_preconditions(
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    Tensor& max,
    Tensor& max_indices) {
  ET_CHECK_SAME_DTYPE2(in, max);
  ET_CHECK_SAME_SHAPE2(max, max_indices);

  // Only support Long as dtype for max_indices.
  ET_CHECK_MSG(
      max_indices.scalar_type() == ScalarType::Long,
      "dtype of the max_indices Tensor expected to be be long.");
  // Ensure dim is valid
  ET_CHECK_VALID_DIM(dim, in.dim());
  ET_CHECK_NON_ZERO_DIM_SIZE(dim, in);
  const auto expected_dim = compute_reduced_out_dim(in, dim, keepdim);
  ET_CHECK_MSG(
      max.dim() == expected_dim && max_indices.dim() == expected_dim,
      "Number of dims of out tensor is not compatible with inputs and params");
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(in);
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(max);
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(max_indices);
}

} // namespace

std::tuple<Tensor&, Tensor&> max_out(
    RuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    Tensor& max,
    Tensor& max_indices) {
  (void)ctx;

  check_preconditions(in, dim, keepdim, max, max_indices);

  Error err_max = resize_reduction_out(in, dim, keepdim, max);
  ET_CHECK_MSG(err_max == Error::Ok, "Failed to resize max tensor in max_out");

  Error err_max_indices = resize_tensor(max_indices, max.sizes());
  ET_CHECK_MSG(
      err_max_indices == Error::Ok,
      "Failed to resize max_indices tensor in max_out");

  dim = dim < 0 ? dim + in.dim() : dim;

  ET_SWITCH_REAL_TYPES_AND(Bool, in.scalar_type(), ctx, "max", CTYPE, [&]() {
    CTYPE* max_data = max.mutable_data_ptr<CTYPE>();
    long* max_indices_data = max_indices.mutable_data_ptr<long>();

    for (size_t out_ix = 0; out_ix < max.numel(); ++out_ix) {
      std::tuple<CTYPE, long> acc = reduce_over_dim<CTYPE>(
          [](CTYPE v, long ix, CTYPE acc_val, long acc_ix) {
            if (!std::isnan(acc_val) && (std::isnan(v) || v > acc_val)) {
              acc_val = v;
              acc_ix = ix;
            }
            return std::tuple<CTYPE, long>{acc_val, acc_ix};
          },
          in,
          dim,
          out_ix);
      max_data[out_ix] = std::get<0>(acc);
      max_indices_data[out_ix] = std::get<1>(acc);
    }
  });
  return {max, max_indices};
}

} // namespace native
} // namespace executor
} // namespace torch
