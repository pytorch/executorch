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

::executorch::aten::Tensor& requantize_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& in_scale_t,
    const ::executorch::aten::Tensor& in_zero_point_t,
    const ::executorch::aten::Tensor& out_scale_t,
    const ::executorch::aten::Tensor& out_zero_point_t,
    const ::executorch::aten::ScalarType out_dtype,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& requantize_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    double in_scale,
    int64_t in_zero_point,
    double out_scale,
    int64_t out_zero_point,
    const ::executorch::aten::ScalarType out_dtype,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
