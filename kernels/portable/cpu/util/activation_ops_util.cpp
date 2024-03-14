/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>

namespace torch {
namespace executor {

bool check_gelu_args(const Tensor& in, string_view approximate, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(in.scalar_type() != ScalarType::Bool);
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      approximate == "tanh" || approximate == "none",
      "Invalid approximation format: %.*s for gelu",
      static_cast<int>(approximate.length()),
      approximate.data());
  return true;
}

bool check_glu_args(const Tensor& in, int64_t dim, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(dim_is_valid(dim, in.dim()));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_floating_type(in));

  const size_t non_negative_dim = dim < 0 ? dim + in.dim() : dim;
  const size_t dim_size = in.size(non_negative_dim);

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      dim_size % 2 == 0,
      "Halving dimension must be even, but dimension %zd is size %zd",
      non_negative_dim,
      dim_size);

  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_floating_type(out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_rank(in, out));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      out.size(non_negative_dim) == dim_size / 2,
      "output tensor must have half the size of the input tensor along the specified dimension.");

  for (size_t i = 0; i < in.dim(); ++i) {
    if (i != non_negative_dim) {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          out.size(i) == in.size(i),
          "output tensor must have the same size as the input tensor in all dimensions except for the specified dimension.");
    }
  }

  return true;
}

bool check_log_softmax_args(
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      !half_to_float, "half to float conversion is not supported on CPU");
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(in));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(out));
  return true;
}

bool check_softmax_args(
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  return check_log_softmax_args(in, dim, half_to_float, out);
}

Error resize_glu_out(const Tensor& in, int64_t dim, Tensor& out) {
  exec_aten::SizesType expected_output_size[kTensorDimensionLimit];

  const size_t non_negative_dim = dim < 0 ? dim + in.dim() : dim;
  for (size_t i = 0; i < in.dim(); i++) {
    expected_output_size[i] =
        (i == non_negative_dim) ? (in.size(i) / 2) : in.size(i);
  }

  ArrayRef<exec_aten::SizesType> output_size{
      expected_output_size, static_cast<size_t>(out.dim())};

  return resize_tensor(out, output_size);
}

} // namespace executor
} // namespace torch
