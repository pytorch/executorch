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
void layer_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    CTYPE eps,
    Tensor& out,
    Tensor& mean,
    Tensor& rstd) {
  const CTYPE* input_data = input.const_data_ptr<CTYPE>();
  const CTYPE* weight_data = weight.const_data_ptr<CTYPE>();
  const CTYPE* bias_data = bias.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
  CTYPE* mean_data = mean.mutable_data_ptr<CTYPE>();
  CTYPE* rstd_data = rstd.mutable_data_ptr<CTYPE>();

  size_t dim = input.size(input.dim() - 1);

  size_t leading_dim = getLeadingDims(input, input.dim() - 1);

  for (int i = 0; i < leading_dim; ++i) {
    const CTYPE* x = input_data + i * dim;
    CTYPE* y = out_data + i * dim;

    // compute E[X] and Var[x] = E[x^2] - E[x]^2
    CTYPE sum = reduce_add(x, dim);
    CTYPE sq_sum = vec_powerf(x, dim);
    CTYPE mean_value = sum / dim;
    CTYPE variance = sq_sum / dim - mean_value * mean_value;
    CTYPE std = std::sqrt(variance + eps);

    // Calculate the elements of output
    for (int j = 0; j < dim; ++j) {
      y[j] = (x[j] - mean_value) / std * weight_data[j] + bias_data[j];
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
    RuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& rstd_out) {
  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, rstd_out);

  ET_KERNEL_CHECK(
      ctx,
      check_layer_norm_args(
          input, normalized_shape, weight, bias, out, mean_out, rstd_out),
      InvalidArgument,
      ret_val);

  if (input.sizes() == out.sizes()) {
    ET_KERNEL_CHECK(
        ctx,
        normalized_shape[0] == input.sizes()[input.dim() - 1],
        InvalidArgument,
        ret_val);
  } else {
    // If we need to resize out to support dynamic input shapes, we can't count
    // on normalized_shape matching the shape of the input or output. But we
    // don't need to modify normalized_shape because it's not used in this
    // function besides some checks
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(out, input.sizes()) == Error::Ok,
        InvalidArgument,
        ret_val);
  }

  ET_SWITCH_FLOAT_TYPES(input.scalar_type(), ctx, __func__, CTYPE, [&]() {
    layer_norm<CTYPE>(
        input, weight.value(), bias.value(), eps, out, mean_out, rstd_out);
  });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
