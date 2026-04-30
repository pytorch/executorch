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
#include <executorch/backends/native/core/RuntimeId.h>

#include <executorch/runtime/core/span.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

/**
 * Sentinel for an unresolved step index. JumpFalseStep::dst_step_idx
 * carries this between router-side emit and PCResolver-side patch. An
 * unresolved value at execute time is a router bug (asserted in debug
 * builds in NativeBackend.cpp::execute).
 *
 * See CONTROL_FLOW_DESIGN.md §5, §8.
 */
inline constexpr size_t kUnresolvedStep =
    std::numeric_limits<size_t>::max();

/**
 * Source-PC sentinel meaning "no source PC" (for synthesized steps the
 * router emitted with no underlying instruction, e.g. a host-pool prep).
 */
inline constexpr uint32_t kNoSourcePc =
    std::numeric_limits<uint32_t>::max();

/**
 * One unit of issued work. Carries dense RuntimeIndex (not opaque
 * RuntimeId). See §4.9 of PORTABLE_BACKEND_API_PROPOSAL.md.
 *
 * The Step variant is intentionally extensible. Today's variants:
 *   - ComputeStep    — dispatch a compiled segment on one Engine.
 *   - TransferStep   — cross-runtime byte movement (host↔device).
 *   - JumpFalseStep  — host-side conditional PC mutation
 *                      (CONTROL_FLOW_DESIGN.md §5).
 *   - MoveStep       — host-side EValue assignment
 *                      (CONTROL_FLOW_DESIGN.md §16).
 *
 * Anticipated future variants (see CONTROL_FLOW_DESIGN.md, "Region IR"
 * follow-up):
 *   - RegionStep     — delegate a whole control-flow region
 *                      (cond/loop/while) to a single Engine that opted
 *                      in via region_capabilities(). Acts as one logical
 *                      step from the executor's POV; the Engine
 *                      decides internally whether to do GPU-side
 *                      branching, host-driven sync, etc.
 *
 * When a new variant is added, three places must be extended:
 *   1. NativeBackend.cpp::execute_step (std::visit branches).
 *   2. NativeBackend.cpp::execute (PC walker tracking
 *      observed_signals; the source_pc helper).
 *   3. routers/GreedyRouter.cpp Phase 6 (host_visible_producer,
 *      step_input_values, step_signal lambdas) — see comments at
 *      those sites for extension guidance.
 *
 * std::visit's exhaustive-dispatch rule means the compiler will flag
 * every site that needs updating. Do not add a default-fallthrough
 * case — let the compiler enforce.
 */
struct ComputeStep {
  RuntimeIndex runtime_idx; // dense index into Plan::instances
  CompiledSegment* segment;
  std::vector<EventId> wait_for;
  EventId signal = kNoEvent;

  // Source PC (instruction index in the chain) of the FIRST kernel in
  // this segment. Used by PCResolver to map source-PC jump destinations
  // to step indices. kNoSourcePc for synthesized steps.
  // See CONTROL_FLOW_DESIGN.md §8.
  uint32_t source_pc = kNoSourcePc;
};

struct TransferStep {
  // Both ends are looked up via bindings at execute time.
  uint32_t src_value_id;
  uint32_t dst_value_id;
  RuntimeIndex src_idx; // hot-path; kHostIdx if host
  RuntimeIndex dst_idx;
  std::vector<EventId> wait_for;
  EventId signal = kNoEvent;

  // Source PC of the originating instruction (the producer or consumer
  // KernelCall this transfer serves; for MoveCall, the MoveCall PC; for
  // PredicateLocator-emitted predicate downloads, the JumpFalseCall PC).
  uint32_t source_pc = kNoSourcePc;
};

/**
 * A control-flow step. Reads `pred_value_id` on host, mutates the
 * executor's PC. See CONTROL_FLOW_DESIGN.md §5.
 *
 * Always runs on host (Invariant CF-1). PredicateLocator (§6) guarantees
 * `pred_value_id` is host-resident at execute time.
 *
 * No `signal` field: a JumpFalseStep produces no value. Steps that
 * follow it but consume values the predicate depends on already wait on
 * those values' producing-step signals via their own wait_for; the jump
 * is data-flow transparent.
 */
struct JumpFalseStep {
  // EValue to inspect (Bool scalar or Bool tensor). Always host-resident
  // at execute time per PredicateLocator (CONTROL_FLOW_DESIGN.md §6).
  uint32_t pred_value_id;

  // Resolved destination as a step index into Plan::steps. Filled by
  // PCResolver in a second pass; emitted with kUnresolvedStep.
  size_t dst_step_idx = kUnresolvedStep;

  // Source-PC of the JumpFalseCall's destination (held alongside
  // dst_step_idx for PCResolver and for diagnostics).
  uint32_t unresolved_dst_pc = kNoSourcePc;

  // Precise transitive dependency closure of pred_value_id, as the
  // signals that produce those dependencies. Filled by DepClosure
  // (CONTROL_FLOW_DESIGN.md §7).
  std::vector<EventId> wait_for;

  // Source PC of the originating JumpFalseCall.
  uint32_t source_pc = kNoSourcePc;
};

/**
 * EValue-level move: `values[dst] = values[src]`. Mirrors ET's
 * MoveCall semantics exactly (runtime/executor/method.cpp).
 *
 * NOT a byte copy. After this step, the destination value SHARES
 * storage with the source value (for tensors, the dst's TensorImpl
 * points at src's data). The router emits a MoveStep in two cases:
 *   1. ET's chain contains a MoveCall (HOP output plumbing).
 *   2. (Future) PredicateLocator host-mirror aliasing.
 *
 * Always runs on host. No signal (synchronous EValue assignment).
 *
 * Distinct from TransferStep (which is for cross-runtime byte movement
 * via Engine::upload_from_host / download_to_host) — see
 * CONTROL_FLOW_DESIGN.md §16.
 */
struct MoveStep {
  uint32_t src_value_id;
  uint32_t dst_value_id;
  uint32_t source_pc = kNoSourcePc;
};

using Step = std::variant<ComputeStep, TransferStep, JumpFalseStep, MoveStep>;

} // namespace native
} // namespace backends
} // namespace executorch
