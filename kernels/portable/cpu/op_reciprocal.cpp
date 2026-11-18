/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace {

template <typename T>
T reciprocal(T x) {
  return static_cast<T>(1.0 / static_cast<double>(x));
}

} // namespace

DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(reciprocal_out, reciprocal)

} // namespace native
} // namespace executor
} // namespace torch
