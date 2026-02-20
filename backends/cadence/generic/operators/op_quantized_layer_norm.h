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

::executorch::aten::Tensor& quantized_layer_norm_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& in_scale,
    const ::executorch::aten::Tensor& in_zero_point,
    __ET_UNUSED const ::executorch::aten::IntArrayRef normalized_shape,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    double eps,
    double output_scale,
    int64_t output_zero_point,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantized_layer_norm_per_tensor_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    double in_scale,
    int64_t in_zero_point,
    __ET_UNUSED const ::executorch::aten::IntArrayRef normalized_shape,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    double eps,
    double output_scale,
    int64_t output_zero_point,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
