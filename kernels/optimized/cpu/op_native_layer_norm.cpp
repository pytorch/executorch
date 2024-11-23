/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>
#include <tuple>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <executorch/kernels/optimized/cpu/moments_utils.h>
#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

template <typename CTYPE>
void layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const optional<Tensor>& weight,
    const optional<Tensor>& bias,
    CTYPE eps,
    Tensor& out,
    Tensor& mean,
    Tensor& rstd) {
  using Vec = at::vec::Vectorized<CTYPE>;

  const size_t dim = input.dim() - normalized_shape.size();
  const size_t dim_size = input.size(dim);

  const size_t M = getLeadingDims(input, dim);
  const size_t N = getTrailingDims(input, dim) * dim_size;

  if (M == 0) {
    return;
  }

  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
  CTYPE* mean_data = mean.mutable_data_ptr<CTYPE>();
  CTYPE* rstd_data = rstd.mutable_data_ptr<CTYPE>();

  if (N == 0) {
    for (int i = 0; i < M; ++i) {
      mean_data[i] = static_cast<CTYPE>(0);
      rstd_data[i] = static_cast<CTYPE>(NAN);
    }
    return;
  }

  const CTYPE* input_data = input.const_data_ptr<CTYPE>();
  const CTYPE* gamma_data;
  if (weight.has_value()) {
    gamma_data = weight.value().const_data_ptr<CTYPE>();
  } else {
    gamma_data = nullptr;
  }
  const CTYPE* beta_data;
  if (bias.has_value()) {
    beta_data = bias.value().const_data_ptr<CTYPE>();
  } else {
    beta_data = nullptr;
  }

  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;

  for (size_t i = 0; i < M; ++i) {
    const CTYPE* src_ptr = input_data + i * N;
    CTYPE* dst_ptr = out_data + i * N;

    CTYPE mean_val;
    CTYPE rstd_val;
    std::tie(mean_val, rstd_val) = RowwiseMoments(src_ptr, N);
    rstd_val = CTYPE(1) / std::sqrt(rstd_val + eps);

    const CTYPE scale = rstd_val;
    const CTYPE offset = -rstd_val * mean_val;

    if (gamma_null || beta_null) {
      for (size_t j = 0; j < N; ++j) {
        const CTYPE gamma_v = gamma_null ? CTYPE(1) : gamma_data[j];
        const CTYPE beta_v = beta_null ? CTYPE(0) : beta_data[j];
        dst_ptr[j] = (src_ptr[j] * scale + offset) * gamma_v + beta_v;
      }
    } else {
      at::vec::map3<CTYPE>(
          [scale, offset](Vec x, Vec gamma, Vec beta) {
            return (x * Vec(scale) + Vec(offset)) * gamma + beta;
          },
          dst_ptr,
          src_ptr,
          gamma_data,
          beta_data,
          N);
    }

    mean_data[i] = mean_val;
    rstd_data[i] = rstd_val;
  }
}

} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> opt_native_layer_norm_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, rstd_out);

  ET_KERNEL_CHECK(
      ctx,
      check_layer_norm_args(
          input, normalized_shape, weight, bias, out, mean_out, rstd_out),
      InvalidArgument,
      ret_val);

  Tensor::SizesType mean_rstd_sizes[kTensorDimensionLimit];
  size_t mean_rstd_ndim = 0;
  get_layer_norm_out_target_size(
      input, normalized_shape, mean_rstd_sizes, &mean_rstd_ndim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(mean_out, {mean_rstd_sizes, mean_rstd_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(rstd_out, {mean_rstd_sizes, mean_rstd_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_SWITCH_FLOAT_TYPES(
      input.scalar_type(), ctx, "native_layer_norm.out", CTYPE, [&]() {
        layer_norm<CTYPE>(
            input,
            normalized_shape,
            weight,
            bias,
            eps,
            out,
            mean_out,
            rstd_out);
      });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
