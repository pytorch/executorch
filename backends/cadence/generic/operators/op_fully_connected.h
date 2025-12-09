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

using ::executorch::aten::optional;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

Tensor& fully_connected_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias,
    Tensor& output);

} // namespace native
} // namespace generic
} // namespace impl
