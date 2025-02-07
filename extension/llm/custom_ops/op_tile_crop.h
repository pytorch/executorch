/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

Tensor& tile_crop_out_impl(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t tile_size,
    Tensor& out);

} // namespace native
} // namespace executor
} // namespace torch
