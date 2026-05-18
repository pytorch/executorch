/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>
#include <numeric>

namespace torch {
namespace executor {

/**
 * Scalar layer_norm computation over M rows of N elements each.
 * Computes mean/variance in float, normalizes with (x - mean) / std * gamma +
 * beta. Caller must handle M==0 and N==0 edge cases before calling.
 */
template <typename CTYPE>
inline void layer_norm_scalar(
    const CTYPE* input_data,
    const CTYPE* weight_data, // nullable
    const CTYPE* bias_data, // nullable
    CTYPE* out_data,
    CTYPE* mean_data,
    CTYPE* rstd_data,
    size_t M,
    size_t N,
    float eps) {
  for (size_t i = 0; i < M; ++i) {
    const CTYPE* x = input_data + i * N;
    CTYPE* y = out_data + i * N;

    // compute E[X] and Var[x] = E[x^2] - E[x]^2
    float sum = std::accumulate(x, x + N, 0.0f);
    float sq_sum = 0;
    for (size_t j = 0; j < N; ++j) {
      sq_sum += static_cast<float>(x[j]) * x[j];
    }
    float mean_value = sum / N;
    float variance = sq_sum / N - mean_value * mean_value;
    float std = std::sqrt(variance + eps);

    // Calculate the elements of output
    for (size_t j = 0; j < N; ++j) {
      CTYPE w = weight_data ? weight_data[j] : static_cast<CTYPE>(1);
      CTYPE b = bias_data ? bias_data[j] : static_cast<CTYPE>(0);
      y[j] = (x[j] - mean_value) / std * w + b;
    }

    mean_data[i] = mean_value;
    rstd_data[i] = 1.0 / std;
  }
}

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
    Tensor& var_out);

bool check_layer_norm_args(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out);

void get_layer_norm_out_target_size(
    const Tensor& in,
    IntArrayRef normalized_shape,
    Tensor::SizesType* mean_rstd_sizes,
    size_t* mean_rstd_ndim);

bool check_group_norm_args(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out);

} // namespace executor
} // namespace torch
