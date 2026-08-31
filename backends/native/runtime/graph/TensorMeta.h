// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// Logical tensor metadata: element type and shape. No storage, no quant scheme.
//
// sizes holds concrete extents. The wire format carries a per-dim range instead,
// but a runtime that plans and executes at fixed shapes cannot honor a dynamic
// dim, so deserialization rejects one rather than silently collapsing it to its
// upper bound.
//
// dim_order_hint is a permutation of dim indices, outermost first; empty means
// contiguous ([0, 1, ..., n-1]). It is a hint only for a tensor with no stored
// content — an activation — where an engine is free to pick its own physical
// layout. For a tensor whose bytes are serialized it instead describes the
// layout those bytes are actually in, and an engine that ignores it reads the
// weight wrong.
struct TensorMeta {
  ScalarType dtype = ScalarType::Float;
  std::vector<int64_t> sizes;
  std::vector<int32_t> dim_order_hint;

  size_t ndim() const {
    return sizes.size();
  }

  // True if dim_order_hint is empty or the identity permutation.
  bool is_contiguous() const;

  // Throws std::runtime_error on a negative extent, or on a count that
  // overflows int64_t.
  int64_t numel() const;

  // Exact on dim_order_hint: an empty hint and a spelled-out identity
  // permutation compare unequal though they mean the same layout.
  bool operator==(const TensorMeta&) const = default;
};

} // namespace ptn
