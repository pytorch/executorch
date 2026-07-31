/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/serialization/schema_generated.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace executorch::backends::webgpu::passes {

// One matched SwiGLU (mul(mul(sigmoid(gate), gate), up) -> out) pattern.
struct SwiGluFusion {
  int common_input_id;
  int gate_id;
  int up_id;
  int sigmoid_id;
  int silu_id;
  int out_id;
  unsigned gate_op;
  unsigned sigmoid_op;
  unsigned mul1_op;
  unsigned mul2_op;
};

// Phase 2: scan fb_graph's op chain for the exact q4-gate/up + sigmoid + 2x
// mul SwiGLU pattern, skipping any op index already in `claimed_ops`.
// Populates `fusions` and the per-op index maps WebGPUGraph::build's Phase 3
// walk uses to fire the fused dispatch, and adds every op index a matched
// pattern owns to `claimed_ops` so a later fusion pass (e.g. QKV-BK64)
// doesn't also try to claim them.
void detect_swiglu_fusions(
    const WebGPUGraph& graph,
    const vkgraph::VkGraph* fb_graph,
    int num_vals,
    std::vector<SwiGluFusion>& fusions,
    std::unordered_map<unsigned, size_t>& gate_producers,
    std::unordered_map<unsigned, size_t>& anchors,
    std::unordered_set<unsigned>& skipped_ops,
    std::unordered_set<unsigned>& claimed_ops);

// Emits the single fused silu_mul_fused dispatch for a matched pattern and
// registers its dynamic-resize hook.
void add_silu_mul_fused_dispatch(
    WebGPUGraph& graph,
    int common_input_id,
    int gate_id,
    int up_id,
    int out_id);

} // namespace executorch::backends::webgpu::passes
