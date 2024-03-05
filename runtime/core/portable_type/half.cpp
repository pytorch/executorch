/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/portable_type/half.h>
#include <ostream>
#include <type_traits>

namespace torch {
namespace executor {

static_assert(
    std::is_standard_layout_v<torch::executor::Half>,
    "Half must be standard layout.");

std::ostream& operator<<(
    std::ostream& out,
    const torch::executor::Half& value) {
  out << (float)value;
  return out;
}

} // namespace executor
} // namespace torch
