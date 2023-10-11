/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>

namespace torch {
namespace executor {

bool check_gelu_args(const Tensor& in, string_view approximate, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      approximate == "tanh" || approximate == "none",
      "Invalid approximation format: %.*s for gelu",
      static_cast<int>(approximate.length()),
      approximate.data());
  return true;
}

bool check_log_softmax_args(
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      !half_to_float, "half to float conversion is not supported on CPU");
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(in));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(out));
  return true;
}

bool check_softmax_args(
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  return check_log_softmax_args(in, dim, half_to_float, out);
}

} // namespace executor
} // namespace torch
