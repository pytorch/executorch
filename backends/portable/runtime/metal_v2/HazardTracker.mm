/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "HazardTracker.h"

#include <algorithm>

namespace executorch {
namespace backends {
namespace metal_v2 {

// Half-open interval overlap: [a.lo, a.hi) and [b.lo, b.hi) overlap iff
// a.lo < b.hi && b.lo < a.hi.
static inline bool rangesOverlap(
    size_t a_lo, size_t a_hi, size_t b_lo, size_t b_hi) {
  return a_lo < b_hi && b_lo < a_hi;
}

void HazardTracker::insertMerged(
    std::vector<Range>& v, size_t lo, size_t hi) {
  if (lo >= hi) {
    return;  // empty range — no-op
  }
  // Walk the (already-sorted) vector. Collapse any range that touches
  // or overlaps [lo, hi) into the new range, drop it from v, and keep
  // going. At the end insert the (possibly grown) merged range at the
  // right sorted position.
  // Two ranges merge when they overlap OR are directly adjacent
  // (touching at a single point), e.g. [0, 64) and [64, 128) collapse
  // into [0, 128) — a common pattern for per-row tile writes.
  size_t merged_lo = lo;
  size_t merged_hi = hi;
  auto write = v.begin();
  for (auto read = v.begin(); read != v.end(); ++read) {
    if (read->hi < merged_lo || read->lo > merged_hi) {
      // Disjoint (not even touching): keep as-is.
      if (write != read) *write = *read;
      ++write;
    } else {
      // Overlapping or adjacent: fold into the merged range.
      merged_lo = std::min(merged_lo, read->lo);
      merged_hi = std::max(merged_hi, read->hi);
    }
  }
  v.erase(write, v.end());

  // Insert at sorted position. Note: parent_mtl is identical for all
  // ranges in v; we borrow it from the first element (or from the
  // caller's range if v was empty before this insert). Rather than
  // threading parent_mtl through this helper, the per-parent writers
  // map's KEY is the parent_mtl, and the map entry's Range values
  // carry it alongside (redundant but keeps Range self-contained for
  // the debug snapshot accessor).
  // Since insertMerged is only called with v = writers_[parent_mtl],
  // the stored ranges all share the same parent. We reuse the parent
  // from v.front() if non-empty; if empty, caller must supply it — so
  // this signature is a helper for the non-empty / merged case, and
  // the public entry points handle the empty case separately.

  Range r;
  r.parent_mtl = v.empty() ? nil : v.front().parent_mtl;
  r.lo = merged_lo;
  r.hi = merged_hi;

  // Sorted insert keyed on lo. v is kept disjoint-and-sorted.
  auto pos = std::lower_bound(
      v.begin(), v.end(), merged_lo,
      [](const Range& x, size_t key) { return x.lo < key; });
  v.insert(pos, r);
}

bool HazardTracker::overlapsAny(
    const std::vector<Range>& v, size_t lo, size_t hi) {
  // v is sorted-and-disjoint. Binary-search the first range whose lo
  // could overlap; scan forward from there. In practice v has few
  // entries so this is dominated by the first comparison.
  if (v.empty()) return false;
  auto it = std::lower_bound(
      v.begin(), v.end(), lo,
      [](const Range& x, size_t key) { return x.hi <= key; });
  // it points to the first range with hi > lo; check if its lo < hi
  // (query). Subsequent ranges are even further right — since v is
  // disjoint we only need to check this one.
  if (it == v.end()) return false;
  return rangesOverlap(it->lo, it->hi, lo, hi);
}

void HazardTracker::trackInput(
    id<MTLBuffer> parent_mtl, size_t lo, size_t hi) {
  if (!parent_mtl || lo >= hi) return;
  pending_inputs_.push_back({parent_mtl, lo, hi});
}

void HazardTracker::trackOutput(
    id<MTLBuffer> parent_mtl, size_t lo, size_t hi) {
  if (!parent_mtl || lo >= hi) return;
  pending_outputs_.push_back({parent_mtl, lo, hi});
}

void HazardTracker::notifyExternalWrite(
    id<MTLBuffer> parent_mtl, size_t lo, size_t hi) {
  if (!parent_mtl || lo >= hi) return;
  auto& bucket = writers_[parent_mtl];
  if (bucket.empty()) {
    bucket.push_back({parent_mtl, lo, hi});
  } else {
    insertMerged(bucket, lo, hi);
  }
}

bool HazardTracker::needsBarrierForPending() const {
  // Check all pending inputs against writers for their parent. If any
  // overlap — RAW hazard — we need a barrier.
  for (const auto& in : pending_inputs_) {
    auto it = writers_.find(in.parent_mtl);
    if (it != writers_.end() && overlapsAny(it->second, in.lo, in.hi)) {
      return true;
    }
  }
  // Same check for outputs — any overlap is a WAW hazard.
  for (const auto& out : pending_outputs_) {
    auto it = writers_.find(out.parent_mtl);
    if (it != writers_.end() && overlapsAny(it->second, out.lo, out.hi)) {
      return true;
    }
  }
  return false;
}

void HazardTracker::commitPending(bool barrier_inserted) {
  // Promote every pending output into the writers index (merged per
  // parent). Pending inputs don't become writers; they're consumed.
  for (const auto& out : pending_outputs_) {
    auto& bucket = writers_[out.parent_mtl];
    if (bucket.empty()) {
      bucket.push_back(out);
    } else {
      insertMerged(bucket, out.lo, out.hi);
    }
  }
  pending_inputs_.clear();
  pending_outputs_.clear();

  if (barrier_inserted) ++stats_.inserted;
  else                  ++stats_.skipped;
}

void HazardTracker::reset() {
  pending_inputs_.clear();
  pending_outputs_.clear();
  writers_.clear();
  // Note: stats_ are NOT reset — they're cumulative across the
  // stream's lifetime so tests can observe barrier-skip counts over
  // a full execute.
}

std::vector<HazardTracker::Range> HazardTracker::writersSnapshot() const {
  // Flatten the per-parent index into a single vector for the debug
  // dump. Only called when R8.1.dbg logging is active.
  std::vector<Range> out;
  for (const auto& [parent, v] : writers_) {
    out.insert(out.end(), v.begin(), v.end());
  }
  return out;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
