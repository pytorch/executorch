/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

Tensor& quantized_embedding_xbit_out(
    // TODO Evaluate whether this name is appropriate for an operator that takes
    // non quant input and returns fp output
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    const int64_t weight_quant_min,
    const int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out,
    int weight_nbit);

Tensor& quantized_embedding_xbit_out(
    KernelRuntimeContext& context,
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out,
    int weight_nbit);

Tensor& quantized_embedding_xbit_dtype_out(
    // TODO Evaluate whether this name is appropriate for an operator that takes
    // non quant input and returns fp output
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    const int64_t weight_quant_min,
    const int64_t weight_quant_max,
    const Tensor& indices,
    std::optional<ScalarType> out_dtype,
    Tensor& out,
    int weight_nbit);

Tensor& quantized_embedding_xbit_dtype_out(
    KernelRuntimeContext& context,
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    std::optional<ScalarType> out_dtype,
    Tensor& out,
    int weight_nbit);

} // namespace native
} // namespace executor
} // namespace torch
