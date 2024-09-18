/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch::executor::native {

// Compute the fast Walsh-Hadamard transform
// (https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform)
// of mat along the last dimension (which must be contiguous).
//
// mat.sizes().back() is currently required to be either a power of
// two, or 28 * a power of two.
Tensor& fast_hadamard_transform_out(
    RuntimeContext& ctx,
    const Tensor& mat,
    Tensor& out);
} // namespace torch::executor::native
