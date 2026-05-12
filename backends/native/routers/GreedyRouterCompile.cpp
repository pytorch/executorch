/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/routers/GreedyRouterContext.h>

#include <executorch/runtime/platform/log.h>

#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace native {
namespace router_internal {

namespace {
using ::executorch::runtime::Error;
using ::executorch::runtime::Span;
} // namespace

Error compile_and_emit_steps(RouterContext& ctx) {
  const auto& graph = ctx.graph;
  auto& plan = ctx.plan;

  // ---- 9. Compile each segment with its value remap --------------------
  for (size_t s = 0; s < ctx.segments.size(); ++s) {
    auto& seg = ctx.segments[s];
    Engine* inst = ctx.instances[seg.provider_idx];
    std::vector<uint32_t> ins(
        seg.input_value_ids.begin(), seg.input_value_ids.end());
    std::vector<uint32_t> outs(
        seg.output_value_ids.begin(), seg.output_value_ids.end());
    ET_LOG(
        Debug,
        "[mem] router: compile_segment %zu provider=%d (%s) instructions=%zu "
        "inputs=%zu outputs=%zu remaps=%zu",
        s,
        seg.provider_idx,
        std::string(ctx.providers[seg.provider_idx]->name()).c_str(),
        seg.instruction_indices.size(),
        ins.size(),
        outs.size(),
        ctx.seg_remaps[s].size());
    auto r = inst->compile_segment(
        Span<const uint32_t>(
            seg.instruction_indices.data(), seg.instruction_indices.size()),
        Span<const uint32_t>(ins.data(), ins.size()),
        Span<const uint32_t>(outs.data(), outs.size()),
        Span<const std::pair<uint32_t, uint32_t>>(
            ctx.seg_remaps[s].data(), ctx.seg_remaps[s].size()));
    if (!r.ok())
      return r.error();
    ctx.compiled_segments.push_back(r.get());
  }

  // ---- 10. Reserve graph input/output bindings --------------------------
  // Input/output destination Buffers are pre-allocated by the executor's
  // allocate_buffers() call (post-route) and bound persistently by
  // prebind_owned_buffers. bind_inputs/bind_outputs re-alias the
  // existing Buffer in place each execute via upload_from_host.
  for (size_t i = 0; i < graph.num_input_ids(); ++i) {
    InputBinding ib;
    ib.value_id = graph.input_id(i);
    plan.inputs.push_back(ib);
  }
  for (size_t i = 0; i < graph.num_output_ids(); ++i) {
    OutputBinding ob;
    ob.value_id = graph.output_id(i);
    plan.outputs.push_back(ob);
  }

  // ---- 11. Emit Steps in order: per-segment transfers, then ComputeStep --
  //
  // Each Step gets a signal event allocated on its owning Engine.
  // wait_for is left empty in v1 (executor walks Steps serially; sync
  // backends settle before the next Step starts). The router records
  // the LAST signal seen on each Engine in plan.terminal_events;
  // the executor waits on those at end of execute() instead of
  // calling drain() per Engine.
  ctx.providers_to_instance.assign(plan.providers.size(), nullptr);
  for (size_t p = 0; p < plan.providers.size() && p < ctx.instances.size();
       ++p) {
    ctx.providers_to_instance[p] = ctx.instances[p];
  }
  auto alloc_signal = [&](RuntimeIndex inst_idx) -> EventId {
    Engine* inst = ctx.providers_to_instance[inst_idx];
    EventSlot slot;
    slot.event = inst->make_event();
    slot.owner = inst;
    EventId id = static_cast<EventId>(plan.events.size());
    plan.events.push_back(std::move(slot));
    return id;
  };

  for (size_t s = 0; s < ctx.segments.size(); ++s) {
    // source_pc for steps belonging to this segment: the FIRST kernel
    // index in the segment. Used by PCResolver (CONTROL_FLOW_DESIGN.md
    // §8) to map source-PC jump targets to step indices.
    uint32_t seg_first_pc = ctx.segments[s].instruction_indices.empty()
        ? kNoSourcePc
        : ctx.segments[s].instruction_indices.front();
    for (const auto& xfer : ctx.seg_transfers[s]) {
      TransferStep ts;
      ts.src_value_id = xfer.src_value_id;
      ts.dst_value_id = xfer.dst_value_id;
      ts.src_idx = static_cast<RuntimeIndex>(xfer.src_provider_idx);
      ts.dst_idx = static_cast<RuntimeIndex>(xfer.dst_provider_idx);
      ts.source_pc = seg_first_pc;
      // Transfer is issued on the device side (non-host slot of the pair).
      RuntimeIndex issuing = (ts.src_idx == 0) ? ts.dst_idx : ts.src_idx;
      ts.signal = alloc_signal(issuing);
      ctx.last_signal_per_inst[issuing] = ts.signal;
      plan.steps.emplace_back(std::move(ts));
    }

    ComputeStep cs;
    cs.runtime_idx = static_cast<RuntimeIndex>(ctx.segments[s].provider_idx);
    cs.segment = ctx.compiled_segments[s];
    cs.source_pc = seg_first_pc;
    cs.signal = alloc_signal(cs.runtime_idx);
    ctx.last_signal_per_inst[cs.runtime_idx] = cs.signal;
    plan.steps.emplace_back(std::move(cs));

    for (const auto& xfer : ctx.seg_post_transfers[s]) {
      TransferStep ts;
      ts.src_value_id = xfer.src_value_id;
      ts.dst_value_id = xfer.dst_value_id;
      ts.src_idx = static_cast<RuntimeIndex>(xfer.src_provider_idx);
      ts.dst_idx = static_cast<RuntimeIndex>(xfer.dst_provider_idx);
      ts.source_pc = seg_first_pc;
      RuntimeIndex issuing = (ts.src_idx == 0) ? ts.dst_idx : ts.src_idx;
      ts.signal = alloc_signal(issuing);
      ctx.last_signal_per_inst[issuing] = ts.signal;
      plan.steps.emplace_back(std::move(ts));
    }
  }

  return Error::Ok;
}

} // namespace router_internal
} // namespace native
} // namespace backends
} // namespace executorch
