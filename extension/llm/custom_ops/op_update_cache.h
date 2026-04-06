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
// is_seq_dim_2: when false, expects [batch, seq, heads, head_dim] layout
//               when true, expects [batch, heads, seq, head_dim] layout
Tensor& update_cache_out(
    RuntimeContext& ctx,
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    bool is_seq_dim_2,
    Tensor& output);

// New function that explicitly takes indices
// is_seq_dim_2: when false, expects [batch, seq, heads, head_dim] layout
//               when true, expects [batch, heads, seq, head_dim] layout
Tensor& update_cache_with_indices_out(
    RuntimeContext& ctx,
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    const Tensor& indices,
    bool is_seq_dim_2,
    Tensor& output);
} // namespace native
} // namespace executor
} // namespace torch
