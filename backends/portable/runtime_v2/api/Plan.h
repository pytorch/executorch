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
#include <executorch/backends/portable/runtime_v2/api/Provider.h>
#include <executorch/backends/portable/runtime_v2/api/Step.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Per-Instance Buffer ownership record. Released at ~Plan via
 * owner->release_buffer(buf). The value_id is the (one) graph value this
 * Buffer was allocated for; for shared/aliased intermediates, multiple
 * `OwnedBuffer` records may exist with different value_ids pointing to
 * the same `Buffer*` (memory-plan aliasing).
 */
struct OwnedBuffer {
  Buffer* buf;
  Instance* owner;
  uint32_t value_id;  // bind this id -> buf at LoadedDelegate construction
};

/**
 * Each event slot remembers which Instance created it. Required so the
 * executor can call the right Instance::wait at the host boundary.
 */
struct EventSlot {
  std::unique_ptr<Event> event;
  Instance* owner;
};

/**
 * Per-input/output binding (router output).
 */
struct InputBinding {
  Location loc;
  uint32_t value_id;
};

struct OutputBinding {
  Location loc;
  uint32_t value_id;
};

/**
 * The router synthesizes new value_ids for cross-runtime transfer
 * destinations. Each synthetic id has a "source" graph value_id whose
 * dtype/shape it inherits. The LoadedDelegate constructor walks this
 * list to extend its EValue array and create matching TensorImpls.
 *
 * Per-execute, a TransferStep moves bytes from source_id to new_id.
 * On host-addressable destination runtimes (CPU, Apple Silicon Metal),
 * the destination Buffer for new_id is allocated by the backend's
 * allocate_all using the AllocRequest's host_alias hint — meaning the
 * destination Buffer aliases the source's host_ptr at init. The
 * per-execute upload_from_host then sees host_ptr == host_ptr and
 * skip-if-same returns immediately (no work).
 *
 * On Vulkan/discrete GPUs, the destination Buffer is real VRAM; the
 * per-execute upload_from_host actually copies bytes via vkCmdCopyBuffer.
 */
struct SyntheticValueDesc {
  uint32_t new_id;
  uint32_t source_id;
};

/**
 * Frozen output of Router::route. Holds:
 *  - Provider/Instance arrays indexed by RuntimeIndex,
 *  - the issue-ordered Step list,
 *  - per-input/output bindings,
 *  - pre-allocated event slots,
 *  - owned-buffer ledger,
 *  - per-provider allocation request lists.
 *
 * See §4.9 of the design doc.
 */
struct Plan {
  // Parallel arrays indexed by RuntimeIndex. By convention, index 0 is
  // the host (CPU) Provider.
  std::vector<Provider*> providers;
  std::vector<Instance*> instances;     // non-owning; lifetime is LoadedDelegate

  std::vector<Step> steps;              // ordered by *issue* (not completion)

  std::vector<InputBinding> inputs;
  std::vector<OutputBinding> outputs;

  // Pre-allocated event slots. Each event is reset lazily by its
  // producing Instance immediately before signaling.
  std::vector<EventSlot> events;

  // Released at Plan destruction. Filled by the post-route allocation
  // step (PortableBackend_v2::allocate_buffers), NOT by the router.
  std::vector<OwnedBuffer> owned_buffers;

  // Synthesized value_ids for cross-runtime transfer destinations.
  // Resolved into TensorImpls by the LoadedDelegate constructor after
  // route() returns. Each synthetic value_id always has a TransferStep
  // emitted (intra-segment intermediates do not appear here).
  std::vector<SyntheticValueDesc> synthetic_values;

  // Per-provider list of allocation requests emitted by the router.
  // Allocation itself is performed by a post-route step (host-first
  // single-pass), which lets the host allocate first so the device's
  // requests can carry host_alias hints pointing at already-allocated
  // host buffers (zero-copy alias for cross-runtime synthetics).
  //
  // alloc_plans[runtime_idx] is the request list for that provider.
  std::vector<std::vector<Instance::AllocRequest>> alloc_plans;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
