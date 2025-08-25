/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

#pragma once

namespace torch {
namespace executor {
namespace native {
namespace utils {

using Scalar = executorch::aten::Scalar;
using Tensor = executorch::aten::Tensor;

Tensor& mul_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out);

Tensor& mul_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out);

} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
