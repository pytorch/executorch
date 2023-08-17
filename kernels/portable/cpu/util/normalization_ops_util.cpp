/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

bool check_batch_norm_args(
    const Tensor& in,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps,
    Tensor& out) {
  // All tensors must be the same dtype
  ET_LOG_AND_RETURN_IF_FALSE(
      tensors_have_same_dtype(in, running_mean, running_var));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  if (weight.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, weight.value()));
  }
  if (bias.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, bias.value()));
  }

  size_t C_dim = in.dim() >= 1 ? 1 : 0;
  // All parameter tensors must be of dim 1 and have length equal to the
  // channels dim of in
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(running_mean, 1));
  ET_LOG_AND_RETURN_IF_FALSE(
      tensors_have_same_size_at_dims(running_mean, 0, in, C_dim));
  if (weight.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(weight.value(), 1));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_size_at_dims(weight.value(), 0, in, C_dim));
  }
  if (bias.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(bias.value(), 1));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_size_at_dims(bias.value(), 0, in, C_dim));
  }

  return true;
}

bool check_layer_norm_args(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out) {
  ET_LOG_AND_RETURN_IF_FALSE(normalized_shape.size() == 1);
  ET_LOG_AND_RETURN_IF_FALSE(weight.has_value());
  if (weight.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(input, weight.value()));
  }
  if (bias.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(input, bias.value()));
  }
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(input, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(input, mean_out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(input, rstd_out));
  ET_LOG_AND_RETURN_IF_FALSE(input.dim() == out.dim());
  return true;
}

} // namespace executor
} // namespace torch
