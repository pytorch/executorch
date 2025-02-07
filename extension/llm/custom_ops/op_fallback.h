/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

namespace native {
Tensor& fallback_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out);
} // namespace native
} // namespace executor
} // namespace torch
