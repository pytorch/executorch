/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace generic {
namespace native {

::executorch::aten::Tensor& quantized_fully_connected_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    int64_t in_zero_point,
    const ::executorch::aten::Tensor& weight_zero_point_t,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    int64_t out_zero_point,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& offset,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantized_fully_connected_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& offset,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor&
quantized_fully_connected_asym8sxasym8s_asym8s_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& offset,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor&
quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& offset,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
