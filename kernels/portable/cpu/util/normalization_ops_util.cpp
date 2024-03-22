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
    Tensor& out,
    Tensor& mean_out,
    Tensor& var_out) {
  // All tensors must be the same dtype
  ET_LOG_AND_RETURN_IF_FALSE(
      tensors_have_same_dtype(in, running_mean, running_var));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, mean_out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, var_out));
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
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(running_var, 1));
  ET_LOG_AND_RETURN_IF_FALSE(
      tensors_have_same_size_at_dims(running_var, 0, in, C_dim));
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
    const Tensor& in,
    IntArrayRef normalized_shape,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out) {
  size_t ndim = normalized_shape.size();
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one element.");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      in.dim() >= ndim,
      "Expected input tensor to have rank >= the length of normalized_shape.");
  size_t shift = in.dim() - ndim;
  for (size_t d = 0; d < ndim; ++d) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        in.size(d + shift) == normalized_shape[d],
        "Expected normalized_shape to match the sizes of input's rightmost dimensions.");
  }
  exec_aten::SizesType shape[ndim];
  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = static_cast<exec_aten::SizesType>(normalized_shape[i]);
  }

  if (weight.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, weight.value()));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensor_has_expected_size(weight.value(), {shape, ndim}));
  }
  if (bias.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, bias.value()));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensor_has_expected_size(bias.value(), {shape, ndim}));
  }
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, mean_out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, rstd_out));
  return true;
}

void get_layer_norm_out_target_size(
    const Tensor& in,
    IntArrayRef normalized_shape,
    Tensor::SizesType* mean_rstd_sizes,
    size_t* mean_rstd_ndim) {
  *mean_rstd_ndim = in.dim();

  for (size_t d = 0; d < in.dim(); ++d) {
    if (d < in.dim() - normalized_shape.size()) {
      mean_rstd_sizes[d] = in.size(d);
    } else {
      mean_rstd_sizes[d] = 1;
    }
  }
}

bool check_group_norm_args(
    const Tensor& in,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out) {
  ET_LOG_AND_RETURN_IF_FALSE(in.size(0) == N);
  ET_LOG_AND_RETURN_IF_FALSE(in.size(1) == C);
  ET_LOG_AND_RETURN_IF_FALSE(in.numel() == N * C * HxW);
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      group > 0, "Expected number of groups to be greater than 0");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      C % group == 0,
      "Expected number of channels in input to be divisible by number of groups");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      !weight.has_value() ||
          (weight.value().dim() == 1 && weight.value().size(0) == C),
      "Expected weight to be a vector of size equal to the number of channels in input");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      !bias.has_value() ||
          (bias.value().dim() == 1 && bias.value().size(0) == C),
      "Expected bias to be a vector of size equal to the number of channels in input");

  if (weight.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, weight.value()));
  }
  if (bias.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, bias.value()));
  }
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, mean_out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, rstd_out));
  return true;
}

} // namespace executor
} // namespace torch
