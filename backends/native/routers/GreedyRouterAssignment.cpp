/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/routers/GreedyRouterContext.h>

#include <executorch/backends/native/core/OpDescriptor.h>
#include <executorch/runtime/platform/log.h>

#include <string>

namespace executorch {
namespace backends {
namespace native {
namespace router_internal {

namespace {

using ::executorch::backends::portable::Graph;
using ::executorch::backends::portable::InstructionKind;
using ::executorch::backends::portable::JumpFalseInfo;
using ::executorch::backends::portable::MoveInfo;
using ::executorch::backends::portable::OperatorCall;
using ::executorch::backends::portable::ValueType;
using ::executorch::runtime::Error;
using ::executorch::runtime::Span;

// Pick a provider for an op. Priority: any provider at index >= 1 that
// returns true from can_run wins (host-slot CPU is the fallback).
int pick_provider(const OperatorCall& op, Span<Runtime* const> providers) {
  OpDescriptor desc;
  desc.name = op.name() ? op.name() : "";

  // Try non-host providers first (index 1..N).
  for (size_t i = 1; i < providers.size(); ++i) {
    if (providers[i]->can_run(desc)) {
      return static_cast<int>(i);
    }
  }
  // Fallback: host (index 0) if available.
  if (!providers.empty() && providers[0]->can_run(desc)) {
    return 0;
  }
  return -1;
}

} // namespace

Error assign_and_segment(RouterContext& ctx) {
  const Graph& graph = ctx.graph;

  // ---- 1. Per-instruction provider assignment ---------------------------
  //
  // Walks every instruction in the chain. For Kernel instructions, picks
  // a provider via greedy preference (any non-host provider that accepts
  // wins; CPU is the fallback). For non-Kernel instructions
  // (JumpFalseCall / MoveCall / FreeCall), records them in
  // `control_instrs` for Phase 5 emission and marks the assignment slot
  // as kSkip. DelegateCall is rejected (CONTROL_FLOW_DESIGN.md §11.1).
  //
  // assignments[] is kept parallel to graph instruction indices so the
  // segmenter (Phase 2) can use index arithmetic.
  ctx.assignments.reserve(graph.num_instructions());
  for (uint32_t i = 0; i < graph.num_instructions(); ++i) {
    InstructionKind kind = graph.instruction_kind(i);
    if (kind == InstructionKind::Delegate) {
      ET_LOG(
          Error,
          "GreedyRouter: DelegateCall at instr %u — nested delegates "
          "inside the v2 chain are not supported (partitioner contract).",
          i);
      return Error::NotSupported;
    }
    if (kind != InstructionKind::Kernel) {
      ctx.assignments.push_back(kSkipNonKernel);
      ControlInstr ci;
      ci.source_pc = i;
      ci.kind = kind;
      if (kind == InstructionKind::JumpFalse) {
        ci.jf = graph.get_jump_false(graph.main_chain_idx(), i);
        ET_LOG(
            Debug,
            "GreedyRouter: instr %u JumpFalseCall(cond=%u, dst=%u)",
            i,
            ci.jf.cond_value_id,
            ci.jf.destination_pc);
      } else if (kind == InstructionKind::Move) {
        ci.mv = graph.get_move(graph.main_chain_idx(), i);
        ET_LOG(
            Debug,
            "GreedyRouter: instr %u MoveCall(src=%u, dst=%u)",
            i,
            ci.mv.src_value_id,
            ci.mv.dst_value_id);
      } else {
        ET_LOG(Debug, "GreedyRouter: instr %u FreeCall (ignored)", i);
      }
      ctx.control_instrs.push_back(ci);
      continue;
    }

    OperatorCall op = graph.get_instruction(i);
    int p = pick_provider(op, ctx.providers);
    if (p < 0) {
      ET_LOG(
          Error,
          "GreedyRouter: no provider for op '%s'",
          op.name() ? op.name() : "?");
      return Error::NotSupported;
    }
    ET_LOG(
        Debug,
        "GreedyRouter: instr %u op='%s' -> provider %d (%s)",
        i,
        op.name() ? op.name() : "?",
        p,
        std::string(ctx.providers[p]->name()).c_str());
    ctx.assignments.push_back(p);
  }

  // ---- 2. Group consecutive same-runtime instructions into segments -----
  //
  // CF-2 (CONTROL_FLOW_DESIGN.md §2): segments must NEVER contain a
  // non-Kernel instruction. Every kSkipNonKernel slot is a hard segment
  // break — consecutive kernels around a jump never merge.
  for (uint32_t i = 0; i < ctx.assignments.size(); ++i) {
    if (ctx.assignments[i] == kSkipNonKernel)
      continue; // CF-2 break
    bool start_new = ctx.segments.empty() ||
        ctx.segments.back().provider_idx != ctx.assignments[i]
        // Previous instruction was a non-Kernel break.
        || (i > 0 && ctx.assignments[i - 1] == kSkipNonKernel);
    if (start_new) {
      ctx.segments.push_back({ctx.assignments[i], {i}, {}, {}});
    } else {
      ctx.segments.back().instruction_indices.push_back(i);
    }
    OperatorCall op = graph.get_instruction(i);
    auto& seg = ctx.segments.back();
    for (size_t j = 0; j < op.num_inputs(); ++j) {
      uint32_t v = op.input(j);
      if (seg.output_value_ids.count(v) == 0)
        seg.input_value_ids.insert(v);
    }
    for (size_t j = 0; j < op.num_outputs(); ++j) {
      seg.output_value_ids.insert(op.output(j));
    }
  }
  ET_LOG(
      Debug,
      "GreedyRouter: built %zu segments, %zu control instructions",
      ctx.segments.size(),
      ctx.control_instrs.size());

  // ---- 3. Build value_id -> producing-segment-index map ----------------
  //
  // value_producer_seg maps value_id → segment index. This is the
  // canonical "who produced this value?" ledger used by DepClosure
  // (Phase 6) and home-provider assignment (Phase 5).
  //
  // EXTENSIBILITY: today values are produced by segments only. With
  // future Region IR (CONTROL_FLOW_DESIGN.md), a value could be
  // produced by an opaque Region (whose internal segment structure
  // the router doesn't introspect). At that point this map's value
  // type may need to widen from "segment_idx" to a tagged union of
  // {segment_idx, region_idx} — or we keep it as segment_idx and add
  // a parallel value_producer_region map. Either is fine; the lambda
  // sites that consume this map (Phase 5 home assignment, Phase 6
  // DepClosure) are already step-kind-aware and would just gain
  // another producer-source case.
  for (size_t s = 0; s < ctx.segments.size(); ++s) {
    for (uint32_t v : ctx.segments[s].output_value_ids) {
      ctx.value_producer_seg[v] = s;
    }
  }

  // ---- 4. Collect graph IO and value-meta -------------------------------
  for (size_t i = 0; i < graph.num_input_ids(); ++i) {
    uint32_t v = graph.input_id(i);
    ctx.io_ids.insert(v);
    ctx.graph_input_ids.insert(v);
  }
  for (size_t i = 0; i < graph.num_output_ids(); ++i) {
    uint32_t v = graph.output_id(i);
    ctx.io_ids.insert(v);
    ctx.graph_output_ids.insert(v);
  }

  // Runtime that consumes each value (could be multiple).
  for (auto& seg : ctx.segments) {
    for (uint32_t v : seg.input_value_ids) {
      ctx.value_consumer_providers[v].insert(seg.provider_idx);
    }
  }

  return Error::Ok;
}

} // namespace router_internal
} // namespace native
} // namespace backends
} // namespace executorch
