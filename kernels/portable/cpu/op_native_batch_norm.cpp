/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <tuple>
#include <utility>

#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>
#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using SizesType = executorch::aten::SizesType;

namespace {

template <typename CTYPE>
auto channel_moments(
    const CTYPE* input,
    size_t batches,
    size_t channels,
    size_t inner) {
  using COMPUTE_T =
      typename executorch::runtime::promote_types<CTYPE, CTYPE, true>::type;
  constexpr size_t kLanes = 8;
  const size_t tail_start = inner - inner % kLanes;
  const auto count = static_cast<COMPUTE_T>(batches * inner);
  std::array<COMPUTE_T, kLanes> sums{};
  COMPUTE_T tail_sum = 0;
  for (size_t b = 0; b < batches; ++b) {
    const CTYPE* x = input + b * channels * inner;
    for (size_t j = 0; j < tail_start; j += kLanes) {
      for (size_t lane = 0; lane < kLanes; ++lane) {
        sums[lane] += static_cast<COMPUTE_T>(x[j + lane]);
      }
    }
    for (size_t j = tail_start; j < inner; ++j) {
      tail_sum += static_cast<COMPUTE_T>(x[j]);
    }
  }
  COMPUTE_T mean =
      (std::accumulate(sums.begin(), sums.end(), COMPUTE_T{0}) + tail_sum) /
      count;
  std::array<COMPUTE_T, kLanes> deviations{};
  std::array<COMPUTE_T, kLanes> squares{};
  COMPUTE_T tail_deviation = 0;
  COMPUTE_T tail_square = 0;
  for (size_t b = 0; b < batches; ++b) {
    const CTYPE* x = input + b * channels * inner;
    for (size_t j = 0; j < tail_start; j += kLanes) {
      for (size_t lane = 0; lane < kLanes; ++lane) {
        const COMPUTE_T d = static_cast<COMPUTE_T>(x[j + lane]) - mean;
        deviations[lane] += d;
        squares[lane] += d * d;
      }
    }
    for (size_t j = tail_start; j < inner; ++j) {
      const COMPUTE_T d = static_cast<COMPUTE_T>(x[j]) - mean;
      tail_deviation += d;
      tail_square += d * d;
    }
  }
  const COMPUTE_T d_sum =
      std::accumulate(deviations.begin(), deviations.end(), COMPUTE_T{0}) +
      tail_deviation;
  const COMPUTE_T sq_sum =
      std::accumulate(squares.begin(), squares.end(), COMPUTE_T{0}) +
      tail_square;
  const COMPUTE_T correction = d_sum / count;
  const COMPUTE_T variance =
      std::max((sq_sum - d_sum * correction) / count, COMPUTE_T{0});
  return std::pair{mean + correction, variance};
}

} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_no_training_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& invstd_out) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, invstd_out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(mean_out, {0}) == Error::Ok, InvalidArgument, ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(invstd_out, {0}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      check_batch_norm_args(
          in,
          weight,
          bias,
          running_mean,
          running_var,
          momentum,
          eps,
          out,
          mean_out,
          invstd_out),
      InvalidArgument,
      ret_val);

  // For now, only support the contiguous dim order
  ET_KERNEL_CHECK(
      ctx,
      is_contiguous_dim_order(in.dim_order().data(), in.dim_order().size()),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, out, mean_out, invstd_out),
      InvalidArgument,
      ret_val);

  if (weight.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(in, weight.value()),
        InvalidArgument,
        ret_val);
  }

  if (bias.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(in, bias.value()),
        InvalidArgument,
        ret_val);
  }

  size_t C_dim = in.dim() >= 1 ? 1 : 0;
  size_t C = in.size(C_dim);
  size_t outer = getLeadingDims(in, C_dim);
  size_t inner = getTrailingDims(in, C_dim);

  static constexpr auto name = "native_batch_norm_legit_no_training.out";

  ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&] {
    const CTYPE* in_data = in.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

    const CTYPE* const mean_data = running_mean.const_data_ptr<CTYPE>();
    const CTYPE* const var_data = running_var.const_data_ptr<CTYPE>();

    for (size_t i = 0; i < outer; ++i) {
      for (size_t c = 0; c < C; ++c) {
        CTYPE mean = mean_data[c];
        CTYPE var = var_data[c];
        CTYPE invstd = 1.0 / std::sqrt(var + eps);
        CTYPE weight_val = 1;
        if (weight.has_value()) {
          weight_val = weight.value().const_data_ptr<CTYPE>()[c];
        }
        CTYPE bias_val = 0;
        if (bias.has_value()) {
          bias_val = bias.value().const_data_ptr<CTYPE>()[c];
        }
        for (size_t j = 0; j < inner; ++j) {
          *out_data = (*in_data - mean) * invstd * weight_val + bias_val;
          out_data++;
          in_data++;
        }
      }
    }
  });

  return ret_val;
}

std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    Tensor& running_mean,
    Tensor& running_var,
    bool training,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& invstd_out) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, invstd_out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      training == false,
      InvalidArgument,
      ret_val,
      "Portable kernels only support inference mode!");

  return _native_batch_norm_legit_no_training_out(
      ctx,
      in,
      weight,
      bias,
      running_mean,
      running_var,
      momentum,
      eps,
      out,
      mean_out,
      invstd_out);
}

std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_no_stats_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    bool training,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& invstd_out) {
  (void)ctx;
  (void)training;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, invstd_out);

  ET_KERNEL_CHECK(
      ctx,
      check_batch_norm_args(
          in,
          weight,
          bias,
          std::optional<Tensor>(),
          std::optional<Tensor>(),
          momentum,
          eps,
          out,
          mean_out,
          invstd_out),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      is_contiguous_dim_order(in.dim_order().data(), in.dim_order().size()),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, out, mean_out, invstd_out),
      InvalidArgument,
      ret_val);

  if (weight.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(in, weight.value()),
        InvalidArgument,
        ret_val);
  }

  if (bias.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(in, bias.value()),
        InvalidArgument,
        ret_val);
  }

  ET_KERNEL_CHECK(ctx, in.dim() >= 2, InvalidArgument, ret_val);

  size_t N = in.size(0);
  size_t C = in.size(1);
  size_t inner = getTrailingDims(in, 1);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(mean_out, {static_cast<SizesType>(C)}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(invstd_out, {static_cast<SizesType>(C)}) == Error::Ok,
      InvalidArgument,
      ret_val);

  static constexpr auto name = "_native_batch_norm_legit.no_stats_out";

  ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&] {
    const CTYPE* in_data = in.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    CTYPE* mean_data = mean_out.mutable_data_ptr<CTYPE>();
    CTYPE* invstd_data = invstd_out.mutable_data_ptr<CTYPE>();

    using COMPUTE_T =
        typename executorch::runtime::promote_types<CTYPE, CTYPE, true>::type;
    for (size_t c = 0; c < C; ++c) {
      const auto [mean, var] =
          channel_moments(in_data + c * inner, N, C, inner);
      const COMPUTE_T invstd =
          COMPUTE_T{1} / std::sqrt(var + static_cast<COMPUTE_T>(eps));
      mean_data[c] = mean;
      invstd_data[c] = invstd;
      const COMPUTE_T weight_val = weight.has_value()
          ? static_cast<COMPUTE_T>(weight.value().const_data_ptr<CTYPE>()[c])
          : 1;
      const COMPUTE_T bias_val = bias.has_value()
          ? static_cast<COMPUTE_T>(bias.value().const_data_ptr<CTYPE>()[c])
          : 0;
      for (size_t b = 0; b < N; ++b) {
        const size_t offset = (b * C + c) * inner;
        for (size_t j = 0; j < inner; ++j) {
          out_data[offset + j] =
              (static_cast<COMPUTE_T>(in_data[offset + j]) - mean) * invstd *
                  weight_val +
              bias_val;
        }
      }
    }
  });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
