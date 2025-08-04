/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
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
  return 1.0 / x;
}

} // namespace

DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(reciprocal_out, reciprocal)

} // namespace native
} // namespace executor
} // namespace torch
