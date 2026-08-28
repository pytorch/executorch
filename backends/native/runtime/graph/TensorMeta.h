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

// One tensor dimension as an inclusive range. Static: min == max. Dynamic:
// min < max, or max < 0 for unbounded.
struct Dim {
  int64_t min = 0;
  int64_t max = -1;

  Dim() = default;
  // Implicit so a shape can be written as a plain int list, e.g. {16, 8}.
  // cppcheck-suppress noExplicitConstructor
  /* implicit */ Dim(int64_t extent) : Dim(extent, extent) {}
  // Throws std::runtime_error on a range no shape can have: a negative lower
  // bound, or a bounded upper bound below it. A funnel, not an invariant --
  // min / max stay public and assignable.
  Dim(int64_t min_v, int64_t max_v);

  bool is_static() const {
    return min == max;
  }

  bool operator==(const Dim&) const = default;
};

// Logical tensor metadata: element type and per-dim size ranges. No storage, no
// quant scheme.
//
// dim_order_hint is a permutation of dim indices, outermost first; empty means
// contiguous ([0, 1, ..., n-1]). It is a hint only for a tensor with no stored
// content — an activation — where an engine is free to pick its own physical
// layout. For a tensor whose bytes are serialized it instead describes the
// layout those bytes are actually in, and an engine that ignores it reads the
// weight wrong.
struct TensorMeta {
  ScalarType dtype = ScalarType::Float;
  std::vector<Dim> sizes;
  std::vector<int32_t> dim_order_hint;

  size_t ndim() const {
    return sizes.size();
  }

  bool is_static() const;

  // True if dim_order_hint is empty or the identity permutation.
  bool is_contiguous() const;

  // Element count from each dim's upper bound. Throws std::runtime_error on an
  // unbounded dynamic dim (max < 0), which has no finite count.
  int64_t numel() const;

  // Exact on dim_order_hint: an empty hint and a spelled-out identity
  // permutation compare unequal though they mean the same layout.
  bool operator==(const TensorMeta&) const = default;
};

} // namespace ptn
