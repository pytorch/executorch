/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <array>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>

namespace torch {
namespace executor {

using Tensor = executorch::aten::Tensor;

bool check_batch_norm_args(
    const Tensor& in,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& var_out) {
  // All tensors must be the same dtype
  if (weight.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, weight.value()));
  }
  if (bias.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, bias.value()));
  }
  if (running_mean.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_dtype(in, running_mean.value()));
  }
  if (running_var.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_dtype(in, running_var.value()));
  }
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, mean_out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, var_out));

  size_t C_dim = in.dim() >= 1 ? 1 : 0;
  // All parameter tensors must be of dim 1 and have length equal to the
  // channels dim of in
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
  if (running_mean.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(running_mean.value(), 1));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_size_at_dims(running_mean.value(), 0, in, C_dim));
  }
  if (running_var.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(running_var.value(), 1));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_size_at_dims(running_var.value(), 0, in, C_dim));
  }

  return true;
}

bool check_layer_norm_args(
    const Tensor& in,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out) {
  size_t ndim = normalized_shape.size();
  ET_CHECK_OR_RETURN_FALSE(
      ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one element; ndim = %zu",
      ndim);
  ET_CHECK_OR_RETURN_FALSE(
      in.dim() >= static_cast<ssize_t>(ndim),
      "Expected input tensor to have rank >= the length of normalized_shape; in.dim() = %" ET_PRI_TENSOR_DIM
      ", ndim = %zu",
      in.dim(),
      ndim);
  ET_CHECK_OR_RETURN_FALSE(
      ndim <= kTensorDimensionLimit,
      "Expected normalized shape to have at most %zu dimensions but it had %zu",
      kTensorDimensionLimit,
      ndim);
  size_t shift = in.dim() - ndim;
  for (const auto d : c10::irange(ndim)) {
    ET_CHECK_OR_RETURN_FALSE(
        in.size(d + shift) == normalized_shape[d],
        "Expected normalized_shape to match the sizes of input's rightmost dimensions; in.size(%zu) = %" ET_PRI_TENSOR_SIZE
        ", normalized_shape[%zu] = %" PRId64,
        d + shift,
        in.size(d + shift),
        d,
        normalized_shape[d]);
  }
  std::array<executorch::aten::SizesType, kTensorDimensionLimit> shape;
  for (const auto i : c10::irange(ndim)) {
    shape[i] = static_cast<executorch::aten::SizesType>(normalized_shape[i]);
  }

  if (weight.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, weight.value()));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensor_has_expected_size(weight.value(), {shape.data(), ndim}));
  }
  if (bias.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, bias.value()));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensor_has_expected_size(bias.value(), {shape.data(), ndim}));
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

  for (const auto d : c10::irange(in.dim())) {
    if (d < static_cast<long>(in.dim() - normalized_shape.size())) {
      mean_rstd_sizes[d] = in.size(d);
    } else {
      mean_rstd_sizes[d] = 1;
    }
  }
}

bool check_group_norm_args(
    const Tensor& in,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
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
  ET_CHECK_OR_RETURN_FALSE(
      group > 0,
      "Expected number of groups to be greater than 0; group = %" PRId64,
      group);
  ET_CHECK_OR_RETURN_FALSE(
      C % group == 0,
      "Expected number of channels in input to be divisible by number of groups; C = %" PRId64
      ", group = %" PRId64 ", C %% group = %" PRId64,
      C,
      group,
      C % group);
  ET_CHECK_OR_RETURN_FALSE(
      !weight.has_value() ||
          (weight.value().dim() == 1 && weight.value().size(0) == C),
      "Expected weight to be a vector of size equal to the number of channels in input; weight.has_value() = %d, weight.dim() = %" ET_PRI_TENSOR_DIM
      ", weight.size(0) = %" ET_PRI_TENSOR_SIZE ", C = %" PRId64,
      weight.has_value(),
      weight.has_value() ? weight.value().dim() : -1,
      weight.has_value() ? weight.value().size(0) : -1,
      C);
  ET_CHECK_OR_RETURN_FALSE(
      !bias.has_value() ||
          (bias.value().dim() == 1 && bias.value().size(0) == C),
      "Expected bias to be a vector of size equal to the number of channels in input; bias.has_value() = %d, bias.dim() = %" ET_PRI_TENSOR_DIM
      ", bias.size(0) = %" ET_PRI_TENSOR_SIZE ", C = %" PRId64,
      bias.has_value(),
      bias.has_value() ? bias.value().dim() : -1,
      bias.has_value() ? bias.value().size(0) : -1,
      C);

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
