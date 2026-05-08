/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// HazardTracker — range-based hazard analysis for the typed-setter
// dispatch path on MetalStream.
// What it tracks
// --------------
// Per dispatch boundary, the stream tells the tracker:
//   - which (parent_mtl, [lo, hi)) ranges are READ inputs (trackInput)
//   - which (parent_mtl, [lo, hi)) ranges are WRITE outputs (trackOutput)
//   - which CPU-side memcpys (host→GPU) just happened (notifyExternalWrite)
// At dispatch time, the stream asks needsBarrierForPending() which
// returns true iff any pending input or output overlaps any prior writer
// range with the same parent_mtl. After dispatch, commitPending() promotes
// pending outputs into the writers index and clears pending.
// Encoder boundary (flush / wait / endEncoder) → reset() drops everything
// — the encoder commit semantics handle ordering across flush windows.
// Why this lives in its own class
// -------------------------------
//   1. Separation of concerns: hazard analysis is a self-contained piece
//      of math over a few vectors.
//   2. Data structure: the writers index is grouped per-parent and
//      merged-on-insert, so the common case (a handful of parents, one
//      or two writes each) is O(1)-amortized per dispatch and the worst
//      case is O(W·R) where W is the number of merged writer ranges for
//      *one* parent and R is the pending input/output count.
//   3. Testability: a unit test for merge correctness only has to
//      construct a HazardTracker, no Metal device required.
// Threading: NOT thread-safe by design. MetalStream is single-threaded.

#include <executorch/backends/metal/core/MetalTypes.h>

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

class HazardTracker {
 public:
  // A half-open byte interval [lo, hi) into a specific MTLBuffer.
  // parent_mtl is held WITHOUT retain — owned upstream by BufferRegistry.
  // Subregion entries pass the parent's MTLBuffer here so two views into
  // the same workspace correctly detect overlap.
  struct Range {
    id<MTLBuffer> parent_mtl;  // borrowed
    size_t lo;
    size_t hi;
  };

  // Counters for dispatches that received vs skipped a barrier. Tests
  // verify the optimization is actually skipping when independent.
  // Cumulative across the tracker's lifetime.
  struct BarrierStats {
    uint64_t inserted = 0;
    uint64_t skipped  = 0;
  };

  HazardTracker() = default;
  ~HazardTracker() = default;

  HazardTracker(const HazardTracker&) = delete;
  HazardTracker& operator=(const HazardTracker&) = delete;

  // Per-dispatch tracking. Called from MetalStream::setInput / setOutput
  // / setInOut / notifyExternalWrite once the (parent_mtl, lo, hi) range
  // has been resolved through BufferRegistry (so Subregion entries are
  // already mapped to their parent + offset).
  void trackInput(id<MTLBuffer> parent_mtl, size_t lo, size_t hi);
  void trackOutput(id<MTLBuffer> parent_mtl, size_t lo, size_t hi);

  // Notify of a host-side write (e.g. CPU memcpy via mps_memcpy or
  // metal_copy_memory). Treated as if it were a prior dispatch's
  // output: future GPU reads of any overlapping range will pre-barrier.
  void notifyExternalWrite(id<MTLBuffer> parent_mtl, size_t lo, size_t hi);

  // Returns true iff any pending input OR pending output range overlaps
  // any prior writer range with the same parent_mtl. Called once per
  // dispatch, before commitPending().
  bool needsBarrierForPending() const;

  // Promote pending outputs into the writers index (merged), clear
  // pending inputs/outputs, bump barrierStats_ accordingly. Called once
  // per dispatch after the GPU work is encoded.
  void commitPending(bool barrier_inserted);

  // Drop all pending and writer state. Called from MetalStream::flush()
  // at encoder boundary — work has been committed and the next encoder
  // doesn't have a hazard with prior dispatches.
  void reset();

  const BarrierStats& barrierStats() const { return stats_; }

  // Debug-only accessors used by MetalCommandRecorder's debug log dump.
  // The vectors returned reflect the ranges as currently stored (pending
  // lists are flat; writers are flattened on demand from the per-parent
  // index).
  const std::vector<Range>& pendingInputs() const { return pending_inputs_; }
  const std::vector<Range>& pendingOutputs() const { return pending_outputs_; }
  std::vector<Range> writersSnapshot() const;

 private:
  // Called by trackInput/trackOutput/notifyExternalWrite to insert a
  // range into the per-parent writers map with on-the-fly merging of
  // adjacent / overlapping ranges. Keeps the writers vector for any
  // single parent O(merged-range-count) instead of O(per-dispatch-write).
  static void insertMerged(std::vector<Range>& v, size_t lo, size_t hi);

  // Returns true iff [lo, hi) overlaps ANY range in v (which is kept
  // sorted-and-merged by insertMerged). Linear scan; in practice v has
  // a small constant number of merged ranges.
  static bool overlapsAny(const std::vector<Range>& v, size_t lo, size_t hi);

  // Pending lists are flat — they're cleared every dispatch boundary
  // and rarely have more than ~4 entries. No need for the per-parent
  // index here.
  std::vector<Range> pending_inputs_;
  std::vector<Range> pending_outputs_;

  // Writers are indexed per parent_mtl with merged ranges. Most ops
  // touch one or two parents (the workspace + maybe a constants buffer),
  // so the map stays tiny (≤ a handful of entries).
  std::map<id<MTLBuffer>, std::vector<Range>> writers_;

  // Readers index. A subsequent dispatch that WRITES to a range that any
  // prior dispatch read needs a WAR barrier (otherwise on MTL4's
  // concurrent dispatch the new write could land before the old read
  // sampled the value). Same merged-range layout as writers_.
  std::map<id<MTLBuffer>, std::vector<Range>> readers_;

  BarrierStats stats_;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
