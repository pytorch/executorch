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

bool check__to_dim_order_copy_args(
    const Tensor& input,
    bool non_blocking,
    exec_aten::OptionalArrayRef<int64_t> dim_order,
    Tensor& out);

} // namespace executor
} // namespace torch
