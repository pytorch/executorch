/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
    Tensor& output,
    Tensor& mean,
    Tensor& rstd) {
  const CTYPE* input_data = input.const_data_ptr<CTYPE>();
  CTYPE* output_data = output.mutable_data_ptr<CTYPE>();
  const CTYPE* weight_data = weight.const_data_ptr<CTYPE>();
  const CTYPE* bias_data = bias.const_data_ptr<CTYPE>();

  size_t dim = input.size(input.dim() - 1);

  size_t leading_dim = getLeadingDims(input, input.dim() - 1);

  for (int i = 0; i < leading_dim; ++i) {
    const CTYPE* x = input_data + i * dim;
    CTYPE* y = output_data + i * dim;

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
  }

  // Assign NAN to mean and rstd. They are not used in seen examples.
  // Use NAN to make the error more obvious in case they are used.
  mean.mutable_data_ptr<CTYPE>()[0] = NAN;
  rstd.mutable_data_ptr<CTYPE>()[0] = NAN;
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
  ET_CHECK_MSG(
      normalized_shape.size() == 1,
      "normalize_shape.size() must be 1 but saw %zd",
      normalized_shape.size());
  ET_CHECK_MSG(weight.has_value(), "Missing weight tensor");
  ET_CHECK_MSG(
      input.scalar_type() == out.scalar_type(),
      "out and input must have the same type.");
  ET_CHECK_MSG(
      input.dim() == out.dim(),
      "out and input must have the same number of dimensions");
  ET_CHECK_MSG(
      input.scalar_type() == mean_out.scalar_type(),
      "mean_out and input must have the same type.");
  ET_CHECK_MSG(
      input.scalar_type() == rstd_out.scalar_type(),
      "rstd_out and input must have the same type.");

  if (input.sizes() == out.sizes()) {
    ET_CHECK_MSG(
        normalized_shape[0] == input.sizes()[input.dim() - 1],
        "Normalized shape value must match the size of input.");
  } else {
    // If we need to resize out to support dynamic input shapes, we can't count
    // on normalized_shape matching the shape of the input or output. But we
    // don't need to modify normalized_shape because it's not used in this
    // function besides some checks
    torch::executor::Error err = resize_tensor(out, input.sizes());
    ET_CHECK_MSG(
        err == torch::executor::Error::Ok,
        "Failed to resize out Tensor in native_layer_norm_out");
  }

// helper for generating the cases for different data types
#define LAYER_NORM(ctype, dtype)                                            \
  case ScalarType::dtype:                                                   \
    layer_norm<ctype>(                                                      \
        input, weight.value(), bias.value(), eps, out, mean_out, rstd_out); \
    break;

  switch (input.scalar_type()) {
    // TODO support bfloat16
    ET_FORALL_FLOAT_TYPES(LAYER_NORM)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", input.scalar_type());
  }
#undef LAYER_NORM
  return {out, mean_out, rstd_out};
}

} // namespace native
} // namespace executor
} // namespace torch
