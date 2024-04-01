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

Error select_copy_util(
    const Tensor& in,
    int64_t dim,
    int64_t index,
    Tensor& out);

} // namespace executor
} // namespace torch
