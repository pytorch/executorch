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

// Logical tensor metadata: element type and bounded shape. No storage, no quant
// scheme. `sizes` holds allocation upper bounds; `lower_bounds` is empty for a
// static shape and holds corresponding lower bounds for a dynamic shape.
//
// dim_order_hint is a permutation of dim indices, outermost first; empty means
// contiguous ([0, 1, ..., n-1]). It is a hint only for a tensor with no stored
// content — an activation — where an engine is free to pick its own physical
// layout. For a tensor whose bytes are serialized, and for a graph input or
// user output whose bytes cross the engine interface, it instead describes the
// layout those bytes are actually in, and an engine that ignores it reads them
// wrong.
struct TensorMeta {
  ScalarType dtype = ScalarType::Float;
  // Upper-bound extents used for allocation.
  std::vector<int64_t> sizes;
  std::vector<int32_t> dim_order_hint;
  // Empty for static shapes. Otherwise one lower bound per extent in `sizes`.
  std::vector<int64_t> lower_bounds;

  size_t ndim() const {
    return sizes.size();
  }

  // True if dim_order_hint is empty or the identity permutation.
  bool is_contiguous() const;

  bool accepts_sizes(const std::vector<int64_t>& candidate) const;

  // Throws std::runtime_error on a negative extent, or on a count that
  // overflows int64_t.
  int64_t numel() const;

  // Exact on dim_order_hint: an empty hint and a spelled-out identity
  // permutation compare unequal though they mean the same layout.
  bool operator==(const TensorMeta&) const = default;
};

} // namespace ptn
