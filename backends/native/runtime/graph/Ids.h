// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace ptn {

// Index-arena handles: a NodeId indexes the graph's node arena, a ValueId
// its value arena. Plain int32_t aliases — they index, compare, and hash
// directly, at the cost of no NodeId/ValueId type distinction. kInvalid marks
// "no id".
using NodeId = int32_t;
using ValueId = int32_t;
inline constexpr int32_t kInvalid = -1;

constexpr bool valid(int32_t id) {
  return id >= 0;
}

// std::cmp_less compares the signed id against the unsigned size without
// casting either side.
constexpr bool in_bounds(int32_t id, size_t size) {
  return valid(id) && std::cmp_less(id, size);
}

} // namespace ptn
