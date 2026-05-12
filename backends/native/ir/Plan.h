/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Engine.h>
#include <executorch/backends/native/core/Event.h>
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
 * Frozen output of Router::route. Holds:
 *  - Runtime/Engine arrays indexed by RuntimeIndex,
 *  - the issue-ordered Step list,
 *  - per-input/output bindings,
 *  - pre-allocated event slots,
 *  - per-provider allocation request lists.
 *
 * Buffer ownership lives in the Engines themselves (each engine releases
 * everything it allocated in its destructor). NativeBackend tracks no
 * per-vid storage state; engines are self-sufficient via their own
 * value->Buffer tables and via set_io_bindings at init.
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

  // Maximum control-flow hops the executor walks before declaring a
  // malformed back-edge. Stamped from RouterOptions::max_hops at
  // route() time so different programs can carry different limits.
  size_t max_hops = 10'000'000;

  // Pre-allocated event slots. Each event is reset lazily by its
  // producing Engine immediately before signaling.
  std::vector<EventSlot> events;

  // EventIds whose signal must complete before execute() returns.
  // Replaces a blanket per-Engine drain. Populated by the router as
  // the signals of the last step on each Engine.
  std::vector<EventId> terminal_events;

  // Mirror identities are conveyed entirely through AllocRequest:
  // device-side mirror requests carry host_mirror_value_id, and
  // engines materialize their own TensorImpls for mirror_ids when they
  // claim the AllocRequest. NativeBackend has no separate mirror table.

  // Per-provider list of allocation requests emitted by the router.
  // Allocation itself is performed by a post-route step in NativeBackend
  // (the bid auction). Engines claim requests they want; HostPool is the
  // fallback claimant for any unclaimed HostMirror.
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
