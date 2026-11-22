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
#include <utility>
#include <vector>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch::backends::cuda {

struct PrivateRowSpan {
  int32_t base{0};
  int32_t capacity{0};
};

struct PrivateRowLayout {
  // Stable owner indices. The last entry is one scratch row. Unassigned arena
  // space amortizes admissions; it does not change any owner's capacity.
  std::vector<PrivateRowSpan> spans;
  int32_t rows{0};
  bool repacked{false};
};

// Plan without touching live payloads. Normal growth keeps unrelated spans
// fixed and chooses destinations disjoint from every old source. Compaction
// and fragmentation fallback pack live spans and require a fresh allocation.
inline runtime::Result<PrivateRowLayout> make_private_row_layout(
    runtime::Span<const int64_t> required_rows,
    runtime::Span<const PrivateRowSpan> previous,
    int64_t initial_rows,
    int64_t limit,
    bool compact = false) {
  constexpr int64_t max_rows = std::numeric_limits<int32_t>::max();
  if (initial_rows <= 0 || limit <= 0 || limit > max_rows ||
      required_rows.size() >= static_cast<size_t>(max_rows) ||
      (!previous.empty() && previous.size() != required_rows.size() + 1)) {
    return runtime::Error::InvalidArgument;
  }
  const int64_t arena_limit = std::min<int64_t>(
      max_rows, static_cast<int64_t>(required_rows.size()) * limit + 1);
  const int64_t old_rows = previous.empty()
      ? 0
      : static_cast<int64_t>(previous[required_rows.size()].base) + 1;
  if (!previous.empty() &&
      (old_rows <= 0 || old_rows > arena_limit ||
       previous[required_rows.size()].capacity != 1)) {
    return runtime::Error::InvalidArgument;
  }
  struct Occupied {
    int64_t base;
    int64_t end;
    size_t owner;
  };
  std::vector<Occupied> occupied;
  std::vector<int64_t> capacities(required_rows.size());
  int64_t packed_rows = 0;
  for (size_t owner = 0; owner < required_rows.size(); ++owner) {
    const int64_t required = required_rows[owner];
    const int64_t old = previous.empty() ? 0 : previous[owner].capacity;
    const int64_t base = previous.empty() ? 0 : previous[owner].base;
    if (required < 0 || required > limit || old < 0 || old > limit ||
        base < 0 || (!previous.empty() && base + old >= old_rows)) {
      return runtime::Error::InvalidArgument;
    }
    if (old > 0) {
      occupied.push_back({base, base + old, owner});
    }
    int64_t capacity = 0;
    if (required > 0) {
      capacity = old > 0 && !compact ? old : std::min(initial_rows, limit);
      if (required > capacity) {
        capacity = std::min(limit, std::max(required, capacity * 2));
      }
    }
    if (capacity > arena_limit - packed_rows - 1) {
      return runtime::Error::InvalidArgument;
    }
    capacities[owner] = capacity;
    packed_rows += capacity;
  }
  std::sort(occupied.begin(), occupied.end(), [](const auto& a, const auto& b) {
    return a.base < b.base;
  });
  for (size_t i = 1; i < occupied.size(); ++i) {
    if (occupied[i].base < occupied[i - 1].end) {
      return runtime::Error::InvalidArgument;
    }
  }
  PrivateRowLayout layout;
  layout.spans.resize(required_rows.size() + 1);
  if (compact || previous.empty()) {
    int64_t base = 0;
    for (size_t owner = 0; owner < capacities.size(); ++owner) {
      layout.spans[owner] = {
          static_cast<int32_t>(base), static_cast<int32_t>(capacities[owner])};
      base += capacities[owner];
    }
    layout.rows = static_cast<int32_t>(base + 1);
    layout.spans.back() = {static_cast<int32_t>(base), 1};
    if (!previous.empty()) {
      layout.repacked = layout.rows != old_rows;
      for (size_t owner = 0; owner < capacities.size(); ++owner) {
        layout.repacked |= layout.spans[owner].base != previous[owner].base;
      }
    }
    return layout;
  }
  int64_t end = 0;
  for (size_t owner = 0; owner < capacities.size(); ++owner) {
    const int64_t capacity = capacities[owner];
    const auto old = previous[owner];
    int64_t base = old.base;
    if (capacity > old.capacity) {
      // Extending an unwrapped owner's span in place preserves its payload.
      bool extends = old.capacity > 0 && base + capacity < arena_limit;
      for (const auto& range : occupied) {
        if (range.owner != owner && base < range.end &&
            base + capacity > range.base) {
          extends = false;
        }
      }
      if (!extends) {
        // Old sources remain occupied until the entire transaction commits.
        std::sort(
            occupied.begin(), occupied.end(), [](const auto& a, const auto& b) {
              return a.base < b.base;
            });
        base = 0;
        for (const auto& range : occupied) {
          if (base + capacity <= range.base) {
            break;
          }
          base = std::max(base, range.end);
        }
      }
      if (base + capacity >= arena_limit) {
        // All owners can fit by capacity; fragmentation is not admission
        // failure. A packed replacement can overlap old sources safely.
        return make_private_row_layout(
            required_rows, previous, initial_rows, limit, true);
      }
      occupied.push_back({base, base + capacity, owner});
    }
    layout.spans[owner] = {
        static_cast<int32_t>(base), static_cast<int32_t>(capacity)};
    if (capacity > 0) {
      end = std::max(end, base + capacity);
    }
  }
  int64_t rows = std::max(old_rows, end + 1);
  if (rows > old_rows) {
    rows = std::min(arena_limit, std::max(rows, 2 * (old_rows - 1) + 1));
  }
  layout.rows = static_cast<int32_t>(rows);
  layout.spans.back() = {static_cast<int32_t>(rows - 1), 1};
  return layout;
}

} // namespace executorch::backends::cuda
