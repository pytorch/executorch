/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/routers/GreedyRouterContext.h>

#include <executorch/runtime/platform/log.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace executorch {
namespace backends {
namespace native {
namespace router_internal {

namespace {
using ::executorch::backends::portable::InstructionKind;
using ::executorch::runtime::Error;
} // namespace

// ---- 12. Control-flow integration (CONTROL_FLOW_DESIGN.md §10) ------
//
// Phases:
//   12a. Interleave control instructions (JumpFalseCall / MoveCall) with
//        the existing source-PC-ordered plan.steps. Emit JumpFalseStep
//        with kUnresolvedStep, and host-host TransferStep for MoveCall.
//        PredicateLocator (CONTROL_FLOW_DESIGN.md §6) ensures every
//        predicate is host-resident.
//   12b. DepClosure (CONTROL_FLOW_DESIGN.md §7) — fill JumpFalseStep
//        wait_for with the precise transitive signal closure of the
//        predicate, using value_producer_seg.
//   12c. PCResolver (CONTROL_FLOW_DESIGN.md §8) — second pass mapping
//        source-PC jump destinations to step indices.
Error interleave_control_flow(RouterContext& ctx) {
  const auto& graph = ctx.graph;
  auto& plan = ctx.plan;

  // ---- Phase 12a: PredicateLocator ----------------------------------
  //
  // For each JumpFalseCall, decide whether the predicate value is
  // already host-resident. The criterion: value_home_provider[pred]
  // == 0 (host home), OR the value is a graph input / constant /
  // mutable buffer (also host-homed by default). If it isn't, the
  // router must mint a host mirror and emit a download
  // TransferStep BEFORE the JumpFalseStep.
  //
  // Today most predicates come from CPU-pinned prim ops
  // (executorch_prim::eq.Scalar etc.) so the home is already host.
  // We log + error if the predicate is non-host: this is the
  // PredicateLocator §6.3 case-3 path; a follow-up CL implements
  // mirror minting.
  auto predicate_is_host = [&](uint32_t pred_vid) -> bool {
    if (pred_vid >= graph.num_values())
      return false;
    // Constant / graph IO / no-producer values default to host.
    if (graph.is_constant(pred_vid))
      return true;
    if (ctx.io_ids.count(pred_vid) > 0)
      return true;
    auto pit = ctx.value_producer_seg.find(pred_vid);
    if (pit == ctx.value_producer_seg.end())
      return true; // placeholder
    // Producer is host?
    return ctx.segments[pit->second].provider_idx == 0
        // Or the value's home is host (so a post-Compute download
        // already exists in plan.steps for it).
        || (ctx.value_home_provider.count(pred_vid) > 0 &&
            ctx.value_home_provider[pred_vid] == 0);
  };

  for (const auto& ci : ctx.control_instrs) {
    if (ci.kind == InstructionKind::JumpFalse) {
      if (!predicate_is_host(ci.jf.cond_value_id)) {
        ET_LOG(
            Error,
            "GreedyRouter: JumpFalseCall at pc=%u predicate value_id=%u "
            "is not host-resident. Non-host predicates require "
            "PredicateLocator mirror minting (CONTROL_FLOW_DESIGN.md "
            "§6.3 case 3) — not yet supported.",
            ci.source_pc,
            ci.jf.cond_value_id);
        return Error::NotSupported;
      }
    }
  }

  // ---- Phase 12b: Interleave control instructions --------------------
  //
  // plan.steps is in source-PC order (segment-by-segment). For each
  // source-PC position of a control instruction, insert it AFTER the
  // last step whose source_pc <= ci.source_pc.
  //
  // We rebuild plan.steps via merge.
  std::vector<Step> merged;
  merged.reserve(plan.steps.size() + ctx.control_instrs.size() * 2);
  size_t step_i = 0;
  size_t ctrl_i = 0;
  auto step_pc = [&](const Step& s) -> uint32_t {
    if (auto* cs = std::get_if<ComputeStep>(&s))
      return cs->source_pc;
    if (auto* ts = std::get_if<TransferStep>(&s))
      return ts->source_pc;
    if (auto* jf = std::get_if<JumpFalseStep>(&s))
      return jf->source_pc;
    if (auto* ms = std::get_if<MoveStep>(&s))
      return ms->source_pc;
    return kNoSourcePc;
  };
  while (step_i < plan.steps.size() || ctrl_i < ctx.control_instrs.size()) {
    bool take_step = false;
    if (step_i >= plan.steps.size()) {
      take_step = false;
    } else if (ctrl_i >= ctx.control_instrs.size()) {
      take_step = true;
    } else {
      uint32_t s_pc = step_pc(plan.steps[step_i]);
      uint32_t c_pc = ctx.control_instrs[ctrl_i].source_pc;
      // Existing steps come BEFORE a control instruction at the same
      // PC: a JumpFalseCall / MoveCall logically follows the steps
      // that produced its operands (those steps' source_pc is the
      // segment's first PC, which is <= the control instruction's PC
      // because the segment ended right before the control instr).
      take_step = (s_pc != kNoSourcePc && s_pc <= c_pc);
    }

    if (take_step) {
      merged.push_back(std::move(plan.steps[step_i++]));
    } else {
      const auto& ci = ctx.control_instrs[ctrl_i++];
      if (ci.kind == InstructionKind::JumpFalse) {
        JumpFalseStep jf;
        jf.pred_value_id = ci.jf.cond_value_id;
        jf.dst_step_idx = kUnresolvedStep;
        jf.unresolved_dst_pc = ci.jf.destination_pc;
        jf.source_pc = ci.source_pc;
        // wait_for is filled by Phase 12c (DepClosure) below.
        merged.emplace_back(std::move(jf));
      } else if (ci.kind == InstructionKind::Move) {
        // EValue-level assignment, NOT a byte copy. Distinct from
        // TransferStep (which is for cross-runtime byte movement).
        // See CONTROL_FLOW_DESIGN.md §16.
        MoveStep ms;
        ms.src_value_id = ci.mv.src_value_id;
        ms.dst_value_id = ci.mv.dst_value_id;
        ms.source_pc = ci.source_pc;
        merged.emplace_back(std::move(ms));
      } else if (ci.kind == InstructionKind::Free) {
        // Ignored: v2 memory plan handles deallocation. See
        // CONTROL_FLOW_DESIGN.md §11.1, §15.
        continue;
      }
    }
  }
  plan.steps = std::move(merged);

  // ---- Phase 12c: DepClosure ----------------------------------------
  //
  // For each value_id, determine which step (if any) makes it
  // host-visible: a host ComputeStep that produces it, or a
  // TransferStep with dst on host, or a MoveStep that aliases it
  // from a host-visible source. This is the basis for closure walks.
  //
  // EXTENSIBILITY: when adding a new Step variant (e.g., a future
  // RegionStep that delegates a whole control-flow region to one
  // Engine), three lambdas below need updates:
  //
  //   * host_visible_producer: which value_ids does this step make
  //     visible on host? (For RegionStep on host: all the region's
  //     declared outputs.)
  //   * step_input_values: which value_ids does this step consume?
  //     (For RegionStep: all values referenced by the region's
  //     predicate-cone + branch bodies that originate outside it.)
  //   * step_signal: what signal does this step emit, if any?
  //
  // Update all three at the same time. DepClosure traversal works
  // automatically once these lambdas know about the new variant.
  std::unordered_map<uint32_t, size_t> host_visible_producer;
  for (size_t si = 0; si < plan.steps.size(); ++si) {
    const auto& s = plan.steps[si];
    if (auto* cs = std::get_if<ComputeStep>(&s)) {
      if (cs->runtime_idx != 0)
        continue; // non-host; not directly visible
      // The segment with this source_pc produces these outputs.
      // segments was indexed by segment number, not source_pc; find
      // the segment that starts at cs->source_pc.
      for (const auto& seg : ctx.segments) {
        if (!seg.instruction_indices.empty() &&
            seg.instruction_indices.front() == cs->source_pc) {
          for (uint32_t v : seg.output_value_ids) {
            host_visible_producer[v] = si;
          }
          break;
        }
      }
    } else if (auto* ts = std::get_if<TransferStep>(&s)) {
      if (ts->dst_idx == 0) {
        host_visible_producer[ts->dst_value_id] = si;
      }
    } else if (auto* ms = std::get_if<MoveStep>(&s)) {
      // EValue assignment: dst becomes host-visible, dependent on src.
      host_visible_producer[ms->dst_value_id] = si;
    }
  }

  // For each step, the set of value_ids whose visibility it consumes
  // (so we can recurse). For ComputeStep: its segment.input_value_ids.
  // For TransferStep host-bound: its src_value_id.
  // For MoveStep: its src_value_id.
  auto step_input_values = [&](size_t si) -> std::vector<uint32_t> {
    std::vector<uint32_t> out;
    const auto& s = plan.steps[si];
    if (auto* cs = std::get_if<ComputeStep>(&s)) {
      for (const auto& seg : ctx.segments) {
        if (!seg.instruction_indices.empty() &&
            seg.instruction_indices.front() == cs->source_pc) {
          out.assign(seg.input_value_ids.begin(), seg.input_value_ids.end());
          break;
        }
      }
    } else if (auto* ts = std::get_if<TransferStep>(&s)) {
      out.push_back(ts->src_value_id);
    } else if (auto* ms = std::get_if<MoveStep>(&s)) {
      out.push_back(ms->src_value_id);
    }
    return out;
  };

  auto step_signal = [&](size_t si) -> EventId {
    const auto& s = plan.steps[si];
    if (auto* cs = std::get_if<ComputeStep>(&s))
      return cs->signal;
    if (auto* ts = std::get_if<TransferStep>(&s))
      return ts->signal;
    // MoveStep / JumpFalseStep have no signal (host-synchronous).
    return kNoEvent;
  };

  auto dep_closure = [&](uint32_t pred_vid) -> std::vector<EventId> {
    std::unordered_set<EventId> sigs;
    std::unordered_set<uint32_t> visited;
    std::vector<uint32_t> work = {pred_vid};
    while (!work.empty()) {
      uint32_t v = work.back();
      work.pop_back();
      if (!visited.insert(v).second)
        continue;
      auto pit = host_visible_producer.find(v);
      if (pit == host_visible_producer.end())
        continue;
      EventId sig = step_signal(pit->second);
      if (sig != kNoEvent)
        sigs.insert(sig);
      for (uint32_t in : step_input_values(pit->second))
        work.push_back(in);
    }
    return std::vector<EventId>(sigs.begin(), sigs.end());
  };

  for (auto& s : plan.steps) {
    if (auto* jf = std::get_if<JumpFalseStep>(&s)) {
      jf->wait_for = dep_closure(jf->pred_value_id);
      ET_LOG(
          Debug,
          "[cf] DepClosure pc=%u pred_vid=%u -> %zu signals",
          jf->source_pc,
          jf->pred_value_id,
          jf->wait_for.size());
    }
  }

  // ---- Phase 12d: PCResolver ----------------------------------------
  //
  // Source-PC → step-index map, second pass. Walk steps in order;
  // each step records its lowest source PC. Then backward-fill so
  // that any source PC inherits the step at or after it.
  size_t n_instr = graph.num_instructions();
  std::vector<size_t> pc_to_step(n_instr, kUnresolvedStep);
  for (size_t si = 0; si < plan.steps.size(); ++si) {
    uint32_t pc = step_pc(plan.steps[si]);
    if (pc == kNoSourcePc || pc >= n_instr)
      continue;
    // Keep the earliest step at this PC.
    if (pc_to_step[pc] == kUnresolvedStep || pc_to_step[pc] > si) {
      pc_to_step[pc] = si;
    }
  }
  // Backward fill.
  size_t next_step = plan.steps.size();
  for (size_t pc_rev = 0; pc_rev < n_instr; ++pc_rev) {
    size_t pc = n_instr - 1 - pc_rev;
    if (pc_to_step[pc] != kUnresolvedStep) {
      next_step = pc_to_step[pc];
    } else {
      pc_to_step[pc] = next_step;
    }
  }
  // Resolve every JumpFalseStep.
  for (auto& s : plan.steps) {
    if (auto* jf = std::get_if<JumpFalseStep>(&s)) {
      if (jf->unresolved_dst_pc < n_instr) {
        jf->dst_step_idx = pc_to_step[jf->unresolved_dst_pc];
      } else {
        // Destination is past the end → falls off the end of the
        // chain (legitimate: branch-end jumps).
        jf->dst_step_idx = plan.steps.size();
      }
      ET_LOG(
          Debug,
          "[cf] PCResolver pc=%u dst_pc=%u -> step_idx=%zu",
          jf->source_pc,
          jf->unresolved_dst_pc,
          jf->dst_step_idx);
    }
  }

  return Error::Ok;
}

} // namespace router_internal
} // namespace native
} // namespace backends
} // namespace executorch
