// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// One tensor dimension as an inclusive range. Static: min == max. Dynamic:
// min < max, or max < 0 for unbounded. Memory is planned from the upper bound.
struct Dim {
  int64_t min = 0;
  int64_t max = -1;

  Dim() = default;
  // A static dimension: min == max == extent. Implicit so a shape can be
  // written as a plain int list, e.g. sizes = {16, 8}.
  // cppcheck-suppress noExplicitConstructor
  /* implicit */ Dim(int64_t extent) : Dim(extent, extent) {}
  // A range [min_v, max_v] (dynamic when min_v < max_v; max_v < 0 unbounded).
  // Throws std::runtime_error on a range no shape can have: a negative lower
  // bound, or a bounded upper bound below it. This catches a malformed
  // serialized shape at the point it enters the IR rather than letting it
  // surface as a wrong numel() later. It is a funnel, not an invariant --
  // min / max stay public and assignable.
  Dim(int64_t min_v, int64_t max_v);

  bool is_static() const {
    return min == max;
  }

  bool operator==(const Dim&) const = default;
};

// Logical tensor metadata: element type and per-dim size ranges. No storage and
// no quant scheme (deferred). dim_order_hint is a *suggested* memory layout — a
// permutation of dim indices, outermost first; empty means contiguous
// ([0, 1, ..., n-1]). It is advisory only: engines choose their own physical
// layout and may ignore it. TensorMeta stays non-prescriptive about layout.
struct TensorMeta {
  ScalarType dtype = ScalarType::Float;
  std::vector<Dim> sizes;
  std::vector<int32_t> dim_order_hint;

  size_t ndim() const {
    return sizes.size();
  }

  // True if every dimension is static (min == max).
  bool is_static() const;

  // True if dim_order_hint is empty or the identity permutation
  // [0, 1, ..., n-1] (i.e. the hint suggests a contiguous layout).
  bool is_contiguous() const;

  // Element count using each dim's upper bound (its size when static). This is
  // the memory-planning extent. Throws std::runtime_error on an unbounded
  // dynamic dim (max < 0), which has no finite element count.
  int64_t numel() const;

  // e.g. "Float[16,16]" (static), "Float[1..8,16]" (bounded dynamic), or
  // "Float[0..?,16]" (unbounded). Debug aid.
  std::string to_string() const;

  bool operator==(const TensorMeta&) const = default;
};

} // namespace ptn
