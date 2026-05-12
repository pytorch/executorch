/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * Internal shared header for the GreedyRouter translation units.
 *
 * GreedyRouter::route() was split into several sibling .cpp files,
 * each implementing one phase of the routing pipeline. The phases
 * pass state via RouterContext (declared below), which collects
 * every container that used to be a local in route().
 *
 * Phase files (all sibling to GreedyRouter.cpp in routers/):
 *   GreedyRouterAssignment.cpp   — phases 1-4
 *   GreedyRouterPlanning.cpp     — phases 5-6
 *   GreedyRouterAllocs.cpp       — phase 7
 *   GreedyRouterTransfers.cpp    — phase 8
 *   GreedyRouterCompile.cpp      — phases 9-11
 *   GreedyRouterControlFlow.cpp  — phase 12
 *
 * Not part of any public API — strictly implementation-internal.
 */

#include <executorch/backends/native/core/Engine.h>
#include <executorch/backends/native/core/Runtime.h>
#include <executorch/backends/native/ir/GraphTypes.h>
#include <executorch/backends/native/ir/Plan.h>
#include <executorch/backends/native/ir/Step.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

// Forward declare to avoid pulling RouterOptions header.
struct RouterOptions;

namespace router_internal {

// ---- Phase-local types lifted to header level so phases can share. -----

// One per non-Kernel instruction (JumpFalseCall / MoveCall / FreeCall),
// recorded by Phase 1 and consumed by Phase 12.
struct ControlInstr {
  uint32_t source_pc;
  ::executorch::backends::portable::InstructionKind kind;
  ::executorch::backends::portable::JumpFalseInfo jf; // valid iff JumpFalse
  ::executorch::backends::portable::MoveInfo mv; // valid iff Move
};

// A run of consecutive same-runtime kernel instructions. Built by
// Phase 2 from `assignments[]`; consumed by phases 3, 5, 6, 8, 9, 11,
// 12c. The input/output sets are computed during Phase 2 traversal.
struct PendingSegment {
  int provider_idx;
  std::vector<uint32_t> instruction_indices;
  std::set<uint32_t> input_value_ids; // consumed-but-not-produced
  std::set<uint32_t> output_value_ids; // produced
};

// One pending boundary transfer between segments (built by Phase 8,
// flushed into TransferStep entries by Phase 11).
struct PendingTransfer {
  uint32_t src_value_id;
  uint32_t dst_value_id;
  int src_provider_idx;
  int dst_provider_idx;
};

// Sentinel value in `assignments[]` for non-Kernel instruction slots.
// Phase 2 treats these as hard segment breaks.
constexpr int kSkipNonKernel = -2;

// Bundles every cross-phase local that used to live in route(). Owns
// no resources (Plan and Engine pointers are non-owning); destroyed
// when route() returns.
struct RouterContext {
  // ---- Inputs (immutable) ----
  const ::executorch::backends::portable::Graph& graph;
  ::executorch::runtime::Span<Runtime* const> providers;
  ::executorch::runtime::Span<Engine* const> instances;
  const ::executorch::runtime::NamedDataMap* ndm;
  const RouterOptions& options;

  // ---- The Plan being built (output) ----
  Plan& plan;

  // ---- Phase 1 outputs ----
  std::vector<int> assignments;
  std::vector<ControlInstr> control_instrs;

  // ---- Phase 2 outputs ----
  std::vector<PendingSegment> segments;

  // ---- Phase 3 outputs ----
  std::unordered_map<uint32_t, size_t> value_producer_seg;

  // ---- Phase 4 outputs ----
  std::set<uint32_t> io_ids;
  std::unordered_set<uint32_t> graph_input_ids;
  std::unordered_set<uint32_t> graph_output_ids;
  std::unordered_map<uint32_t, std::set<int>> value_consumer_providers;

  // ---- Phase 5 outputs ----
  std::unordered_map<uint32_t, int> value_home_provider;

  // ---- Phase 6 outputs ----
  std::map<std::pair<uint32_t, int>, uint32_t> mirror_table;
  uint32_t next_mirror_id = 0;

  // ---- Phase 8 outputs ----
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> seg_remaps;
  std::vector<std::vector<PendingTransfer>> seg_transfers;
  std::vector<std::vector<PendingTransfer>> seg_post_transfers;
  std::unordered_map<uint32_t, std::vector<size_t>> writers_per_value;
  std::unordered_map<uint32_t, std::vector<size_t>> readers_per_value;

  // ---- Phase 9 outputs ----
  std::vector<CompiledSegment*> compiled_segments;

  // ---- Phase 11 working state (consumed by terminal-events finalize) ----
  std::vector<Engine*> providers_to_instance;
  std::unordered_map<RuntimeIndex, EventId> last_signal_per_inst;
};

// ---- Phase function prototypes -----------------------------------------

// Phases 1-4: per-instruction provider assignment, segment grouping,
// producer map, graph IO + consumer providers.
::executorch::runtime::Error assign_and_segment(RouterContext& ctx);

// Phases 5-6: home planner + mirror planner.
void plan_homes_and_mirrors(RouterContext& ctx);

// Phase 7: const + intermediate alloc emission, DeviceMirror allocs,
// HostExtern allocs for graph IO.
::executorch::runtime::Error emit_allocs(RouterContext& ctx);

// Phase 8: per-segment boundary transfer planning.
void plan_transfers(RouterContext& ctx);

// Phases 9-11: compile each segment, reserve IO bindings, emit ordered
// Steps with signal-event allocation.
::executorch::runtime::Error compile_and_emit_steps(RouterContext& ctx);

// Phase 12: predicate locator, control-flow interleave, dep closure,
// PC resolver. Only called when ctx.control_instrs is non-empty.
::executorch::runtime::Error interleave_control_flow(RouterContext& ctx);

} // namespace router_internal
} // namespace native
} // namespace backends
} // namespace executorch
