/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Event.h>
#include <executorch/backends/native/core/Engine.h>
#include <executorch/backends/native/core/Runtime.h>
#include <executorch/backends/native/core/RuntimeId.h>
#include <executorch/backends/native/ir/Step.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

/**
 * Each event slot remembers which Engine created it. Required so the
 * executor can call the right Engine::wait at the host boundary.
 */
struct EventSlot {
  std::unique_ptr<Event> event;
  Engine* owner;
};

/**
 * Per-input/output binding (router output).
 */
struct InputBinding {
  uint32_t value_id;
};

struct OutputBinding {
  uint32_t value_id;
};

/**
 * Per-Plan record of a value materialized on multiple sides of a runtime
 * boundary.
 *
 * The router emits one MirrorValueDesc whenever it needs a value to
 * exist on more than one runtime simultaneously:
 *   - Cross-runtime intermediates (canonical home is host pool;
 *     non-host runtimes get a `mirror_id` that aliases or copies the
 *     host bytes).
 *   - Producer-side mirrors for values whose home is host but whose
 *     producer is on a non-host segment.
 *
 * `source_value_id` is the program-level / canonical value_id this
 * mirror is a copy of. For cross-runtime intermediates this is the
 * host-homed id.
 *
 * `mirror_id` is the per-runtime id, freshly minted by the router past
 * the end of the original graph value space. NativeBackend uses this
 * (plus value_owner) to bucket per-execute IO bind work to the right
 * engine; the engine's internal value→Buffer table holds the actual
 * Buffer for mirror_id.
 *
 * Per-execute, a TransferStep may move bytes between source and mirror
 * (intermediates only — constants are immutable, so no per-execute
 * transfer is needed). On host-addressable destination runtimes (CPU,
 * Apple Silicon Metal), the mirror's Buffer aliases the host source's
 * storage at allocate_buffers time (UMA collapse on Apple Silicon —
 * single physical allocation; per-execute upload_from_host then sees
 * the alias unchanged and skip-completes).
 *
 * On Vulkan / discrete GPUs, the mirror is a real VRAM Buffer; the
 * per-execute upload_from_host actually copies bytes via
 * vkCmdCopyBuffer.
 */
struct MirrorValueDesc {
  uint32_t mirror_id;
  uint32_t source_value_id;
};

/**
 * Frozen output of Router::route. Holds:
 *  - Runtime/Engine arrays indexed by RuntimeIndex,
 *  - the issue-ordered Step list,
 *  - per-input/output bindings,
 *  - pre-allocated event slots,
 *  - per-provider allocation request lists.
 *
 * Buffer ownership lives in the Engines themselves (each engine releases
 * everything it allocated in its destructor). NativeBackend tracks just
 * one piece of cross-engine bookkeeping — value_owner, vid → which engine
 * claimed it — and that lives on DelegateInstance, not on Plan.
 *
 * See §4.9 of the design doc.
 */
struct Plan {
  // Parallel arrays indexed by RuntimeIndex. By convention, index 0 is
  // the HostPool (canonical home for boundary values; not a compute
  // runtime — its can_run() always returns false). Compute providers
  // (CpuRuntime, MetalRuntime, ...) occupy slots 1+.
  std::vector<Runtime*> providers;
  std::vector<Engine*> instances; // non-owning; lifetime is DelegateInstance

  std::vector<Step> steps; // ordered by *issue* (not completion)

  std::vector<InputBinding> inputs;
  std::vector<OutputBinding> outputs;

  // Pre-allocated event slots. Each event is reset lazily by its
  // producing Engine immediately before signaling.
  std::vector<EventSlot> events;

  // EventIds whose signal must complete before execute() returns.
  // Replaces a blanket per-Engine drain. Populated by the router as
  // the signals of the last step on each Engine.
  std::vector<EventId> terminal_events;

  // Mirror pairs emitted for cross-runtime transfer destinations.
  // Per-engine value→Buffer tables hold the actual Buffer for
  // mirror_id; this list is the lookup index by source_value_id used by
  // NativeBackend's IO bucketing.
  std::vector<MirrorValueDesc> mirror_values;

  // Per-provider list of allocation requests emitted by the router.
  // Allocation itself is performed by a post-route step in NativeBackend
  // (the bid auction). Engines claim requests they want; HostPool is the
  // fallback claimant for any unclaimed HostMirror/HostOnly.
  //
  // alloc_plans[runtime_idx] is the request list for that provider.
  std::vector<std::vector<Engine::AllocRequest>> alloc_plans;

  // Per-provider list of constant upload requests emitted by the router,
  // partitioned by which engines actually consume each constant. The
  // router fills this purely as planning output — engine I/O is driven
  // by NativeBackend's post-route upload_constants pass (symmetric to
  // alloc_plans / materialize_buffers).
  //
  // const_plans[runtime_idx] is the constant-request list for that
  // provider. Each engine independently materializes its constants
  // (zero-copy NDM alias on CPU / Apple-Silicon Metal; device-side
  // load on discrete GPU). No cross-engine coordination is required;
  // multiple engines may hold independent Buffer wrappers for the
  // same NDM key without duplicating bytes (UMA platforms).
  std::vector<std::vector<Engine::ConstRequest>> const_plans;
};

} // namespace native
} // namespace backends
} // namespace executorch
