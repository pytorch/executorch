/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch::backends::webgpu {

// Pure predicate for the slice dual-store dispatch merge; unit-tested by
// slice_dual_guard_test so the load-bearing condition is checked without a GPU.
//
// A chained aten.slice_copy.Tensor is a PURE ELEMENTWISE COPY of its input --
// out2[i] == out1[i] at the identical flat index -- only when it starts at a
// STATIC 0, steps by 1, and its STATIC end already covers the serialized
// maximum extent of the sliced dim. norm_clamp() then lands on the live size at
// every live shape, so the identity holds for the whole dynamic-shape range.
//
// A SymInt bound is rejected: it can resolve below the live extent at execute
// time, at which point the second slice is a strict subset and copying the
// whole gather into it would be wrong.
struct SliceDualSpan {
  int64_t step = 0;
  bool start_is_symint = true;
  bool end_is_symint = true;
  int64_t start = 0; // resolved at the serialized maximum shape
  int64_t end = 0; // resolved at the serialized maximum shape
  int64_t dim_size = 0; // serialized maximum extent of the sliced dim
};

inline bool slice_dual_full_span(const SliceDualSpan& s) {
  if (s.step != 1) {
    return false;
  }
  if (s.start_is_symint || s.end_is_symint) {
    return false;
  }
  if (s.dim_size <= 0) {
    return false;
  }
  return s.start == 0 && s.end >= s.dim_size;
}

// The fused dispatch gathers from `in` and stores to `out1` and `out2` in one
// invocation. `out2` must therefore not be either of the buffers the gather
// already reads or writes.
inline bool
slice_dual_buffers_ok(const void* out2, const void* out1, const void* in) {
  return out2 != nullptr && out2 != out1 && out2 != in;
}

} // namespace executorch::backends::webgpu
