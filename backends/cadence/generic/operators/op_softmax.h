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

::executorch::aten::Tensor& _softmax_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    int64_t dim,
    __ET_UNUSED bool half_to_float,
    ::executorch::aten::Tensor& Y);

::executorch::aten::Tensor& _softmax_f32_f32_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    int64_t dim,
    __ET_UNUSED ::executorch::aten::optional<bool> half_to_float,
    ::executorch::aten::Tensor& Y);

} // namespace native
} // namespace generic
} // namespace impl
