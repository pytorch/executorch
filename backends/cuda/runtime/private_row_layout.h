/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch::backends::cuda {

struct PrivateRowSpan {
  int32_t base{0};
  int32_t capacity{0};
};

struct PrivateRowLayout {
  // Stable owner indices; inactive entries occupy no device rows. The final
  // entry is one scratch row, outside every owner's address range.
  std::vector<PrivateRowSpan> spans;
  int32_t rows{0};
};

// Plan a replacement without mutating the live layout. required_rows is
// already capped by the layer's window for Ring storage. A zero requirement
// reclaims that owner's span; otherwise its capacity never shrinks here.
// The caller fences and copies retained rows before committing this plan.
inline runtime::Result<PrivateRowLayout> make_private_row_layout(
    runtime::Span<const int64_t> required_rows,
    runtime::Span<const PrivateRowSpan> previous,
    int64_t initial_rows,
    int64_t limit) {
  constexpr int64_t max_rows = std::numeric_limits<int32_t>::max();
  if (initial_rows <= 0 || limit <= 0 || limit > max_rows ||
      required_rows.size() >= static_cast<size_t>(max_rows) ||
      (!previous.empty() && previous.size() != required_rows.size() + 1)) {
    return runtime::Error::InvalidArgument;
  }
  // Reject the entire request before allocating or constructing a plan.
  int64_t previous_end = 0;
  for (size_t owner = 0; owner < required_rows.size(); ++owner) {
    if (required_rows[owner] < 0 || required_rows[owner] > limit ||
        (!previous.empty() &&
         (previous[owner].base != previous_end ||
          previous[owner].capacity < 0 || previous[owner].capacity > limit ||
          static_cast<int64_t>(previous[owner].base) +
                  previous[owner].capacity >
              max_rows))) {
      return runtime::Error::InvalidArgument;
    }
    if (!previous.empty()) {
      previous_end += previous[owner].capacity;
    }
  }
  if (!previous.empty() &&
      (previous[required_rows.size()].base != previous_end ||
       previous[required_rows.size()].capacity != 1 ||
       previous_end >= max_rows)) {
    return runtime::Error::InvalidArgument;
  }
  PrivateRowLayout layout;
  layout.spans.reserve(required_rows.size() + 1);
  int64_t base = 0;
  for (size_t owner = 0; owner < required_rows.size(); ++owner) {
    const int64_t required = required_rows[owner];
    int64_t capacity = 0;
    if (required > 0) {
      const int64_t old = previous.empty() ? 0 : previous[owner].capacity;
      capacity = old > 0 ? old : std::min(initial_rows, limit);
      if (required > capacity) {
        capacity = std::min(limit, std::max(required, capacity * 2));
      }
    }
    // Keep space for scratch, and bound GPU Int32 address descriptors.
    if (capacity > max_rows - base - 1) {
      return runtime::Error::InvalidArgument;
    }
    layout.spans.push_back(
        {static_cast<int32_t>(base), static_cast<int32_t>(capacity)});
    base += capacity;
  }
  layout.spans.push_back({static_cast<int32_t>(base), 1});
  layout.rows = static_cast<int32_t>(base + 1);
  return layout;
}

} // namespace executorch::backends::cuda
