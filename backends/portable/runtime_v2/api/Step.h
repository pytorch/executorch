/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Event.h>
#include <executorch/backends/portable/runtime_v2/api/Instance.h>
#include <executorch/backends/portable/runtime_v2/api/Location.h>

#include <executorch/runtime/core/span.h>

#include <cstdint>
#include <variant>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * One unit of issued work. Carries dense RuntimeIndex (not opaque
 * RuntimeId). See §4.9 of the design doc.
 */
struct ComputeStep {
  RuntimeIndex runtime_idx;     // dense index into Plan::instances
  CompiledSegment* segment;
  QueueKind queue;              // Compute in v1
  ::executorch::runtime::Span<const EventId> wait_for;
  EventId signal;               // kNoEvent = none
};

struct TransferStep {
  // Both ends are looked up via bindings at execute time.
  uint32_t src_value_id;
  uint32_t dst_value_id;
  Location src;                 // identity-tagged, for diagnostics & trace
  Location dst;
  RuntimeIndex src_idx;         // hot-path; kHostIdx if host
  RuntimeIndex dst_idx;
  QueueKind queue;              // typically Transfer
  ::executorch::runtime::Span<const EventId> wait_for;
  EventId signal;
};

using Step = std::variant<ComputeStep, TransferStep>;

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
