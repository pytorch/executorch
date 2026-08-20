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

::executorch::aten::Tensor& quantized_add_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    const ::executorch::aten::Tensor& X_scale,
    const ::executorch::aten::Tensor& X_zero_point,
    const ::executorch::aten::Tensor& Y,
    const ::executorch::aten::Tensor& Y_scale,
    const ::executorch::aten::Tensor& Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantized_add_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const ::executorch::aten::Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantized_add_Scalar_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    const ::executorch::aten::Tensor& X_scale,
    const ::executorch::aten::Tensor& X_zero_point,
    const ::executorch::aten::Scalar& Y,
    double out_scale,
    int64_t out_zero_point,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantized_add_asym8sxasym8s_asym8s_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const ::executorch::aten::Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantized_add_asym8uxasym8u_asym8u_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const ::executorch::aten::Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
