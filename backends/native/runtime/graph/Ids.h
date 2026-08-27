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

// Index-arena handles: a NodeRef indexes the graph's node arena, a ValueRef its
// value arena, a GraphRef a subgraph arena (HOP branch bodies). Plain int32_t
// aliases — they index, compare, and hash directly, at the cost of no type
// distinction between them. kInvalid marks "no ref".
using NodeRef = int32_t;
using ValueRef = int32_t;
using GraphRef = int32_t;
inline constexpr int32_t kInvalid = -1;

constexpr bool valid(int32_t ref) {
  return ref >= 0;
}

// std::cmp_less compares the signed ref against the unsigned size without
// casting either side.
constexpr bool in_bounds(int32_t ref, size_t size) {
  return valid(ref) && std::cmp_less(ref, size);
}

} // namespace ptn
