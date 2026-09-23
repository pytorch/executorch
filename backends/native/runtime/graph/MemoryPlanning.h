// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn {

using AllocationId = int32_t;
using MemoryKind = int32_t;

inline constexpr AllocationId kInvalidAllocationId = -1;
inline constexpr MemoryKind kInvalidMemoryKind = -1;

// One engine-owned tensor allocation requirement. `size_bytes` is the required
// storage span from the allocation base, and `alignment` is the minimum base
// alignment. Engines assign `memory_kind` as an opaque compatibility class;
// only requests with the same class may share an allocation.
struct AllocationRequest {
  ValueId value_id = kInvalid;
  size_t size_bytes = 0;
  size_t alignment = 1;
  MemoryKind memory_kind = kInvalidMemoryKind;
};

struct PlannedAllocation {
  size_t size_bytes = 0;
  size_t alignment = 1;
  MemoryKind memory_kind = kInvalidMemoryKind;
  std::vector<ValueId> value_ids;
};

class MemoryPlan {
 private:
  std::vector<PlannedAllocation> allocations_;
  std::vector<AllocationId> allocation_ids_;

  friend MemoryPlan plan_memory(
      const Graph& graph,
      const std::vector<AllocationRequest>& requests);

 public:
  MemoryPlan() = default;

  const std::vector<PlannedAllocation>& allocations() const {
    return allocations_;
  }

  // Returns kInvalidAllocationId for a valid value without a request. Throws
  // if value_id does not address the graph used to construct this plan.
  AllocationId allocation_id(ValueId value_id) const;

  // Throws if allocation_id does not address this plan's allocation list.
  const PlannedAllocation& allocation(AllocationId allocation_id) const;
};

// Greedily assigns requested alias groups in descending size order.
// Allocations can be reused only when memory_kind matches and all inclusive
// lifetimes are strictly disjoint. Requested parameters, constant tensors, and
// buffers remain live for the whole graph. Callers decide which values require
// managed storage. Every member of a requested alias group contributes to its
// lifetime, but only requested values receive allocation ids.
//
// TODO: Represent nonzero alias storage offsets in Graph. Until then, every
// alias is treated as starting at its group's allocation base; size_bytes must
// describe the full required span from that base.
MemoryPlan plan_memory(
    const Graph& graph,
    const std::vector<AllocationRequest>& requests);

} // namespace ptn
