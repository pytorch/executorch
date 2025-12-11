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

::executorch::aten::Tensor& quantized_softmax_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& mask,
    int64_t dim,
    const ::executorch::aten::Tensor& in_scale,
    const ::executorch::aten::Tensor& in_zero_point,
    const ::executorch::aten::Tensor& out_scale,
    const ::executorch::aten::Tensor& out_zero_point,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantized_softmax_per_tensor_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& mask,
    int64_t dim,
    double in_scale,
    int64_t in_zero_point,
    double out_scale,
    int64_t out_zero_point,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
