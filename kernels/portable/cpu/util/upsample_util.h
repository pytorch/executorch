/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

bool check_upsample_2d_common_args(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t>& output_size,
    const executorch::aten::OptionalArrayRef<double>& scale_factors,
    Tensor& out);

bool check_upsample_bilinear2d_args(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t>& output_size,
    const bool align_corners,
    const executorch::aten::OptionalArrayRef<double>& scale_factors,
    Tensor& out);

bool check_upsample_nearest2d_args(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t>& output_size,
    const executorch::aten::OptionalArrayRef<double>& scale_factors,
    Tensor& out);

Error resize_upsample_2d(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t>& output_size,
    const executorch::aten::OptionalArrayRef<double>& scale_factors,
    double& scale_h_out,
    double& scale_w_out,
    Tensor& out);

// Ported from aten/src/ATen/native/UpSample.h
template <typename scalar_t>
inline scalar_t compute_scales_value(
    const std::optional<double>& scale,
    int64_t input_size,
    int64_t output_size) {
  return scale.has_value() ? static_cast<scalar_t>(1.0 / scale.value())
                           : (static_cast<scalar_t>(input_size) / output_size);
}

// Ported from aten/src/ATen/native/UpSample.h
template <typename scalar_t>
inline scalar_t area_pixel_compute_scale(
    int64_t input_size,
    int64_t output_size,
    bool align_corners,
    const std::optional<double>& scale) {
  // see Note [area_pixel_compute_scale]
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<scalar_t>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<scalar_t>(0);
    }
  } else {
    return compute_scales_value<scalar_t>(scale, input_size, output_size);
  }
}

// Ported from aten/src/ATen/native/UpSample.h
template <typename scalar_t>
inline scalar_t area_pixel_compute_source_index(
    scalar_t scale,
    int64_t dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    scalar_t src_idx = scale * (dst_index + static_cast<scalar_t>(0.5)) -
        static_cast<scalar_t>(0.5);
    return (!cubic && src_idx < static_cast<scalar_t>(0)) ? scalar_t(0)
                                                          : src_idx;
  }
}

// Ported from aten/src/ATen/native/UpSample.h
// when `real_input_index` becomes larger than the range the floating point
// type can accurately represent, the type casting to `int64_t` might exceed
// `input_size`, causing overflow. So we guard it with `std::min` below.
template <typename scalar_t, typename opmath_t>
inline void guard_index_and_lambda(
    const opmath_t& real_input_index,
    const int64_t& input_size,
    int64_t& input_index,
    scalar_t& lambda) {
  input_index =
      std::min(static_cast<int64_t>(floorf(real_input_index)), input_size - 1);
  lambda = std::min(
      std::max(real_input_index - input_index, static_cast<opmath_t>(0)),
      static_cast<opmath_t>(1));
}

// Ported from aten/src/ATen/native/UpSample.h
template <typename scalar_t, typename opmath_t>
inline void compute_source_index_and_lambda(
    int64_t& input_index0,
    int64_t& input_index1,
    scalar_t& lambda0,
    scalar_t& lambda1,
    opmath_t ratio,
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    bool align_corners) {
  if (output_size == input_size) {
    // scale_factor = 1, simply copy
    input_index0 = output_index;
    input_index1 = output_index;
    lambda0 = static_cast<scalar_t>(1);
    lambda1 = static_cast<scalar_t>(0);
  } else {
    const auto real_input_index = area_pixel_compute_source_index<opmath_t>(
        ratio, output_index, align_corners, /*cubic=*/false);
    guard_index_and_lambda(real_input_index, input_size, input_index0, lambda1);
    int64_t offset = (input_index0 < input_size - 1) ? 1 : 0;
    input_index1 = input_index0 + offset;
    lambda0 = static_cast<scalar_t>(1.) - lambda1;
  }
}

// Ported from aten/src/ATen/native/UpSample.h
inline int64_t nearest_neighbor_compute_source_index(
    const float scale,
    int64_t dst_index,
    int64_t input_size) {
  // Index computation matching OpenCV INTER_NEAREST
  // which is buggy and kept for BC
  const int64_t src_index =
      std::min(static_cast<int64_t>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}

} // namespace executor
} // namespace torch
