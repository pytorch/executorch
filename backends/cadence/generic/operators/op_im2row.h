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

::executorch::aten::Tensor& im2row_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    ::executorch::aten::IntArrayRef kernel_size,
    ::executorch::aten::IntArrayRef dilation,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef stride,
    const ::executorch::aten::Tensor& in_zero_point,
    bool channel_last,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& im2row_per_tensor_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    ::executorch::aten::IntArrayRef kernel_size,
    ::executorch::aten::IntArrayRef dilation,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef stride,
    int64_t in_zero_point,
    bool channel_last,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
