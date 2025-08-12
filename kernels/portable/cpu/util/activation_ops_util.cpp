/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_shape_to_c_string.h>

namespace torch {
namespace executor {

bool check_gelu_args(
    const Tensor& in,
    std::string_view approximate,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(in.scalar_type() != ScalarType::Bool);
  ET_CHECK_OR_RETURN_FALSE(
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
  const ssize_t dim_size = in.size(non_negative_dim);

  ET_CHECK_OR_RETURN_FALSE(
      dim_size % 2 == 0,
      "Halving dimension must be even, but dimension %zd is size %zd",
      non_negative_dim,
      dim_size);

  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_floating_type(out));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_rank(in, out));
  ET_CHECK_OR_RETURN_FALSE(
      out.size(non_negative_dim) == dim_size / 2,
      "output tensor must have half the size of the input tensor along the specified dimension; out.size(%zu) = %" ET_PRI_TENSOR_SIZE
      ", dim_size = %zd",
      non_negative_dim,
      out.size(non_negative_dim),
      dim_size);

  for (const auto i : c10::irange(in.dim())) {
    if (static_cast<size_t>(i) != non_negative_dim) {
      if (out.size(i) != in.size(i)) {
#if ET_LOG_ENABLED
        auto out_shape_str = executorch::runtime::tensor_shape_to_c_string(
            executorch::runtime::Span<const Tensor::SizesType>(
                out.sizes().data(), out.sizes().size()));
        auto in_shape_str = executorch::runtime::tensor_shape_to_c_string(
            executorch::runtime::Span<const Tensor::SizesType>(
                in.sizes().data(), in.sizes().size()));
        ET_LOG(
            Error,
            "output tensor must have the same size as the input tensor in all dimensions except for the specified dimension. (output shape: %s input shape: %s)",
            out_shape_str.data(),
            in_shape_str.data());
#endif // ET_LOG_ENABLED
        return false;
      }
    }
  }

  return true;
}

bool check_log_softmax_args(
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  ET_CHECK_OR_RETURN_FALSE(
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
  executorch::aten::SizesType expected_output_size[kTensorDimensionLimit];

  const size_t non_negative_dim = dim < 0 ? dim + in.dim() : dim;
  for (const auto i : c10::irange(in.dim())) {
    expected_output_size[i] = (static_cast<size_t>(i) == non_negative_dim)
        ? (in.size(i) / 2)
        : in.size(i);
  }

  ArrayRef<executorch::aten::SizesType> output_size{
      expected_output_size, static_cast<size_t>(out.dim())};

  return resize_tensor(out, output_size);
}

} // namespace executor
} // namespace torch
