/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>
#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>
#include <tuple>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

template <typename CTYPE>
void group_norm(
    const Tensor& input,
    const optional<Tensor>& weight,
    const optional<Tensor>& bias,
    int64_t sN,
    int64_t sC,
    int64_t sHxW,
    int64_t group,
    CTYPE eps,
    Tensor& out,
    Tensor& mean,
    Tensor& rstd) {
  size_t N = static_cast<size_t>(sN); // NOLINT
  size_t C = static_cast<size_t>(sC); // NOLINT
  size_t HxW = static_cast<size_t>(sHxW); // NOLINT
  size_t G = static_cast<size_t>(group); // NOLINT

  size_t leading = N * G;
  size_t D = C / G;
  size_t inner_size = D * HxW;

  if (leading == 0) {
    return;
  }

  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
  CTYPE* mean_data = mean.mutable_data_ptr<CTYPE>();
  CTYPE* rstd_data = rstd.mutable_data_ptr<CTYPE>();

  if (inner_size == 0) {
    for (int i = 0; i < leading; ++i) {
      mean_data[i] = static_cast<CTYPE>(0);
      rstd_data[i] = static_cast<CTYPE>(NAN);
    }
    return;
  }

  const CTYPE* input_data = input.const_data_ptr<CTYPE>();
  const CTYPE* weight_data;
  if (weight.has_value()) {
    weight_data = weight.value().const_data_ptr<CTYPE>();
  } else {
    weight_data = nullptr;
  }
  const CTYPE* bias_data;
  if (bias.has_value()) {
    bias_data = bias.value().const_data_ptr<CTYPE>();
  } else {
    bias_data = nullptr;
  }

  for (int i = 0; i < leading; ++i) {
    const CTYPE* x = input_data + i * inner_size;

    // compute E[X] and Var[x] = E[x^2] - E[x]^2
    CTYPE sum = reduce_add(x, inner_size);
    CTYPE sq_sum = vec_powerf(x, inner_size);
    CTYPE mean_value = sum / inner_size;
    CTYPE variance = sq_sum / inner_size - mean_value * mean_value;
    CTYPE std = std::sqrt(variance + eps);
    CTYPE rstd_value = 1.0 / std;

    // Calculate the elements of output
    if (weight_data == nullptr && bias_data == nullptr) {
      CTYPE* y = out_data + i * inner_size;
      for (size_t j = 0; j < inner_size; j++) {
        y[j] = (x[j] - mean_value) * rstd_value;
      }
    } else {
      const size_t g = i % G;
      for (size_t j = 0; j < D; j++) {
        const size_t ch = g * D + j;
        const CTYPE scale =
            rstd_value * (weight_data == nullptr ? 1.0 : weight_data[ch]);
        const CTYPE beta =
            -scale * mean_value + (bias_data == nullptr ? 0.0 : bias_data[ch]);
        x = input_data + (i * D + j) * HxW;
        CTYPE* y = out_data + (i * D + j) * HxW;
        for (size_t k = 0; k < HxW; k++) {
          y[k] = scale * x[k] + beta;
        }
      }
    }

    mean_data[i] = mean_value;
    rstd_data[i] = rstd_value;
  }
}

} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> native_group_norm_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, rstd_out);

  ET_KERNEL_CHECK(
      ctx,
      check_group_norm_args(
          input, weight, bias, N, C, HxW, group, out, mean_out, rstd_out),
      InvalidArgument,
      ret_val);

  Tensor::SizesType mean_rstd_sizes[kTensorDimensionLimit];
  mean_rstd_sizes[0] = N;
  mean_rstd_sizes[1] = group;

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(mean_out, {mean_rstd_sizes, 2}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(rstd_out, {mean_rstd_sizes, 2}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx, tensor_is_default_dim_order(input), InvalidArgument, ret_val);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(input, out, mean_out, rstd_out),
      InvalidArgument,
      ret_val);

  if (weight.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(input, weight.value()),
        InvalidArgument,
        ret_val);
  }

  if (bias.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(input, bias.value()),
        InvalidArgument,
        ret_val);
  }

  constexpr auto name = "native_group_norm.out";

  ET_SWITCH_FLOAT_TYPES(input.scalar_type(), ctx, name, CTYPE, [&]() {
    group_norm<CTYPE>(
        input, weight, bias, N, C, HxW, group, eps, out, mean_out, rstd_out);
  });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
