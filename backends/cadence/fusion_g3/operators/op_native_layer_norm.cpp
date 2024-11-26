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
#include <xa_nnlib_kernels_api.h>
#include <cmath>
#include <tuple>

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using IntArrayRef = exec_aten::ArrayRef<int64_t>;
using torch::executor::Error;
using torch::executor::KernelRuntimeContext;

namespace cadence {
namespace impl {
namespace G3 {
namespace native {

namespace {

template <typename CTYPE>
void layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    CTYPE eps,
    Tensor& out,
    Tensor& mean,
    Tensor& rstd) {
  size_t dim = input.dim() - normalized_shape.size();
  size_t dim_size = input.size(dim);

  size_t leading = executorch::runtime::getLeadingDims(input, dim);
  size_t normalized =
      executorch::runtime::getTrailingDims(input, dim) * dim_size;

  if (leading == 0) {
    return;
  }

  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
  CTYPE* mean_data = mean.mutable_data_ptr<CTYPE>();
  CTYPE* rstd_data = rstd.mutable_data_ptr<CTYPE>();

  if (normalized == 0) {
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
    const CTYPE* x = input_data + i * normalized;
    CTYPE* y = out_data + i * normalized;

    // compute E[X] and Var[x] = E[x^2] - E[x]^2
    CTYPE sum = torch::executor::reduce_add(x, normalized);
    CTYPE sq_sum = torch::executor::vec_powerf(x, normalized);
    CTYPE mean_value = sum / normalized;
    CTYPE variance = sq_sum / normalized - mean_value * mean_value;
    CTYPE std = std::sqrt(variance + eps);

    // Calculate the elements of output
    for (int j = 0; j < normalized; ++j) {
      CTYPE w = weight_data ? weight_data[j] : static_cast<CTYPE>(1);
      CTYPE b = bias_data ? bias_data[j] : static_cast<CTYPE>(0);
      y[j] = (x[j] - mean_value) / std * w + b;
    }

    mean_data[i] = mean_value;
    rstd_data[i] = 1.0 / std;
  }
}

} // namespace

// native_layer_norm.out(Tensor input, int[] normalized_shape, Tensor? weight,
// Tensor? bias, float eps, *, Tensor(a!) out, Tensor(b!) mean_out, Tensor(c!)
// rstd_out) -> (Tensor(a!), Tensor(b!), Tensor(c!))
// As a reference, there's math_native_layer_norm in ATen:
// https://www.internalfb.com/code/fbsource/[2da5b17b086554c6cd0c3ab08a35aeec2a8bad8c]/xplat/caffe2/aten/src/ATen/native/layer_norm.cpp?lines=188
std::tuple<Tensor&, Tensor&, Tensor&> native_layer_norm_out(
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
      torch::executor::check_layer_norm_args(
          input, normalized_shape, weight, bias, out, mean_out, rstd_out),
      InvalidArgument,
      ret_val);

  // Only support default dim order for now.
  // TODO: Support other dim orders.
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensor_is_default_dim_order(input),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(
          input, out, mean_out, rstd_out),
      InvalidArgument,
      ret_val);

  if (weight.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        executorch::runtime::tensors_have_same_dim_order(input, weight.value()),
        InvalidArgument,
        ret_val);
  }

  if (bias.has_value()) {
    ET_KERNEL_CHECK(
        ctx,
        executorch::runtime::tensors_have_same_dim_order(input, bias.value()),
        InvalidArgument,
        ret_val);
  }
  int kTensorDimensionLimit = executorch::runtime::kTensorDimensionLimit;
  Tensor::SizesType mean_rstd_sizes[kTensorDimensionLimit];
  size_t mean_rstd_ndim = 0;
  torch::executor::get_layer_norm_out_target_size(
      input, normalized_shape, mean_rstd_sizes, &mean_rstd_ndim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(
          mean_out, {mean_rstd_sizes, mean_rstd_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(
          rstd_out, {mean_rstd_sizes, mean_rstd_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  int input_shape[kTensorDimensionLimit];
  for (int i = 0; i < input.dim(); i++) {
    input_shape[i] = input.size(i);
  }

  if (out.scalar_type() == ScalarType::Float) {
    float* const out_data = out.mutable_data_ptr<float>();
    float* const mean_data = mean_out.mutable_data_ptr<float>();
    float* const rstd_data = rstd_out.mutable_data_ptr<float>();
    const float* const inp_data = input.const_data_ptr<float>();
    int dim = input.dim() - normalized_shape.size();

    int num_elm = 1;
    for (int i = 0; i < normalized_shape.size(); i++) {
      num_elm *= normalized_shape[i];
    }

    float* weight_data;
    if (weight.has_value()) {
      weight_data = weight.value().mutable_data_ptr<float>();
    } else {
      weight_data = (float*)malloc(num_elm * sizeof(float));
      for (int i = 0; i < num_elm; i++) {
        weight_data[i] = 1;
      }
    }
    float* bias_data;
    if (bias.has_value()) {
      bias_data = bias.value().mutable_data_ptr<float>();
    } else {
      bias_data = (float*)malloc(num_elm * sizeof(float));
      for (int i = 0; i < num_elm; i++) {
        bias_data[i] = 0;
      }
    }

    xa_nn_native_layer_norm_f32_f32(
        out_data,
        mean_data,
        rstd_data,
        inp_data,
        input_shape,
        input.dim(),
        dim,
        weight_data,
        bias_data,
        (float)eps);

    if (!bias.has_value()) {
      free(bias_data);
    }
    if (!weight.has_value()) {
      free(weight_data);
    }
  } else {
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
  }

  return ret_val;
}

} // namespace native
} // namespace G3
} // namespace impl
} // namespace cadence