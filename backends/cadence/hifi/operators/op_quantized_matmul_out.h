/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/kernel/kernel_runtime_context.h"

namespace impl {
namespace HiFi {
namespace native {

::executorch::aten::Tensor& quantized_matmul_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& X,
    int64_t X_zero_point,
    const ::executorch::aten::Tensor& Y,
    int64_t Y_zero_point,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace HiFi
} // namespace impl
