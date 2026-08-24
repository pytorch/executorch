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

// Index-arena handles. A NodeRef indexes Graph.nodes; a ValueRef indexes
// Graph.values. Plain int32_t aliases — they index, compare, and hash
// directly, at the cost of no NodeRef/ValueRef type distinction. kInvalid
// marks "no ref" (e.g. a graph input has no producer node; a fresh value has
// no alias).
using NodeRef = int32_t;
using ValueRef = int32_t;
constexpr int32_t kInvalid = -1;

inline bool valid(int32_t ref) {
  return ref >= 0;
}

// True when `ref` addresses one of `size` elements: valid and in range. The
// signed/unsigned comparison goes through std::cmp_less so neither side has to
// be cast to the other's signedness.
inline bool in_bounds(int32_t ref, size_t size) {
  return valid(ref) && std::cmp_less(ref, size);
}

} // namespace ptn
