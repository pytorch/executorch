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

::executorch::aten::Tensor& avg_pool2d_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    ::executorch::aten::IntArrayRef kernel_size,
    ::executorch::aten::IntArrayRef stride,
    ::executorch::aten::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    ::executorch::aten::optional<int64_t> divisor_override,
    const ::executorch::aten::optional<::executorch::aten::Tensor>&
        in_zero_point_t,
    bool channel_last,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
