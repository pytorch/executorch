/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/quantized/cpu/embeddingxb.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cinttypes>
#include <cmath>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using Scalar = executorch::aten::Scalar;
using ScalarType = executorch::aten::ScalarType;

/**
 * Retrieves the embeddings specified by indices, dequantizes them, and stores
 * them in out. The weight is quantized per channel, with a scale and zero_point
 * for each embedding.
 *
 * Corresponds as the out variant to torch.ops.quantized.embedding_2bit
 *
 * NOTE: quant_min, quant_max, and Dtype are not used in computation, but rather
 * metadata that is passed around which can be useful for pattern matching. See
 * https://github.com/pytorch/pytorch/pull/87093#discussion_r1000841181 for more
 * info.
 */
Tensor& quantized_embedding_2bit_out(
    // TODO Evaluate whether this name is appropriate for an operator that takes
    // non quant input and returns fp output
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    const int64_t weight_quant_min,
    const int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out) {
  return quantized_embedding_xbit_out(
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out,
      2);
}

Tensor& quantized_embedding_2bit_out(
    KernelRuntimeContext& context,
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out) {
  return quantized_embedding_xbit_out(
      context,
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out,
      2);
}

Tensor& quantized_embedding_2bit_dtype_out(
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    std::optional<ScalarType> out_dtype,
    Tensor& out) {
  return quantized_embedding_xbit_dtype_out(
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out_dtype,
      out,
      2);
}

Tensor& quantized_embedding_2bit_dtype_out(
    KernelRuntimeContext& context,
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    std::optional<ScalarType> out_dtype,
    Tensor& out) {
  return quantized_embedding_xbit_dtype_out(
      context,
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out_dtype,
      out,
      2);
}

} // namespace native
} // namespace executor
} // namespace torch
