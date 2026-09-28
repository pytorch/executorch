/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace torch {
namespace executor {

/**
 * Scalar layer_norm computation over M rows of N elements each.
 * Uses corrected two-pass moments in float (double for Double inputs).
 * Caller must handle M==0 and N==0 edge cases before calling.
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
    double eps) {
  using COMPUTE_T =
      typename executorch::runtime::promote_types<CTYPE, CTYPE, true>::type;
  const auto count = static_cast<COMPUTE_T>(N);
  const auto epsilon = static_cast<COMPUTE_T>(eps);
  // Independent accumulators allow vectorization without reassociating sums.
  constexpr size_t kLanes = 8;
  for (size_t i = 0; i < M; ++i) {
    const CTYPE* x = input_data + i * N;
    CTYPE* y = out_data + i * N;

    std::array<COMPUTE_T, kLanes> sums{};
    size_t j = 0;
    for (; j + kLanes <= N; j += kLanes) {
      for (size_t lane = 0; lane < kLanes; ++lane) {
        sums[lane] += static_cast<COMPUTE_T>(x[j + lane]);
      }
    }
    COMPUTE_T sum = std::accumulate(sums.begin(), sums.end(), COMPUTE_T{0});
    for (; j < N; ++j) {
      sum += static_cast<COMPUTE_T>(x[j]);
    }
    COMPUTE_T mean_value = sum / count;

    std::array<COMPUTE_T, kLanes> deviations{};
    std::array<COMPUTE_T, kLanes> squares{};
    j = 0;
    for (; j + kLanes <= N; j += kLanes) {
      for (size_t lane = 0; lane < kLanes; ++lane) {
        const COMPUTE_T d = static_cast<COMPUTE_T>(x[j + lane]) - mean_value;
        deviations[lane] += d;
        squares[lane] += d * d;
      }
    }
    COMPUTE_T d_sum =
        std::accumulate(deviations.begin(), deviations.end(), COMPUTE_T{0});
    COMPUTE_T sq_sum =
        std::accumulate(squares.begin(), squares.end(), COMPUTE_T{0});
    for (; j < N; ++j) {
      const COMPUTE_T d = static_cast<COMPUTE_T>(x[j]) - mean_value;
      d_sum += d;
      sq_sum += d * d;
    }
    const COMPUTE_T correction = d_sum / count;
    const COMPUTE_T variance =
        std::max((sq_sum - d_sum * correction) / count, COMPUTE_T{0});
    mean_value += correction;
    const COMPUTE_T std = std::sqrt(variance + epsilon);

    // Calculate the elements of output
    for (j = 0; j < N; ++j) {
      const COMPUTE_T w =
          weight_data ? static_cast<COMPUTE_T>(weight_data[j]) : COMPUTE_T{1};
      const COMPUTE_T b =
          bias_data ? static_cast<COMPUTE_T>(bias_data[j]) : COMPUTE_T{0};
      y[j] = (static_cast<COMPUTE_T>(x[j]) - mean_value) / std * w + b;
    }

    mean_data[i] = mean_value;
    rstd_data[i] = COMPUTE_T{1} / std;
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
