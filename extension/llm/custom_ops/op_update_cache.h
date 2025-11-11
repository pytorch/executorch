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

// Original update_cache_out function without indices parameter
Tensor& update_cache_out(
    RuntimeContext& ctx,
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    Tensor& output);

// New function that explicitly takes indices
Tensor& update_cache_with_indices_out(
    RuntimeContext& ctx,
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    const Tensor& indices,
    Tensor& output);
} // namespace native
} // namespace executor
} // namespace torch
