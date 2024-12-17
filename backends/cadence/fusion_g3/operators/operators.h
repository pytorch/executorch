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

namespace cadence {
namespace impl {
namespace G3 {
namespace native {

::executorch::aten::Tensor& add_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Tensor& b,
    const ::executorch::aten::Scalar& alpha,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& add_scalar_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& a,
    const ::executorch::aten::Scalar& b,
    const ::executorch::aten::Scalar& alpha,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace G3
} // namespace impl
} // namespace cadence
