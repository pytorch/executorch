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

// One matched QKV-BK64 (three q4gsw linears sharing one input, exact Llama
// Q/K/V geometry) pattern.
struct QkvBk64Fusion {
  int input_id = -1;
  int output_ids[3] = {-1, -1, -1};
  int weight_ids[3] = {-1, -1, -1};
  int scale_ids[3] = {-1, -1, -1};
  unsigned op_indices[3] = {0, 0, 0};
  size_t separate_begin[3] = {0, 0, 0};
  size_t separate_end[3] = {0, 0, 0};
  size_t fused_dispatch = SIZE_MAX;
  WGPUBuffer params_buffer = nullptr;
  uint32_t max_m = 0;
};

// True if the device meets the BK64 kernel's shader-f16/workgroup limits.
bool qkv_bk64_device_supported(WGPUDevice device);

// Phase 2: scan fb_graph's op chain for three q4gsw-linear ops sharing one
// input in exact Q/K/V geometry. Populates `fusions` and the per-op index
// maps Phase 3 uses; does NOT filter against already-claimed op indices --
// SwiGLU keeps precedence over an overlapping QKV candidate, so call
// retain_unclaimed_qkv_fusions after SwiGLU detection completes.
void detect_qkv_bk64_fusions(
    const WebGPUGraph& graph,
    const vkgraph::VkGraph* fb_graph,
    int num_vals,
    std::vector<QkvBk64Fusion>& fusions,
    std::unordered_map<unsigned, size_t>& first_ops,
    std::unordered_map<unsigned, size_t>& last_ops,
    std::unordered_map<unsigned, size_t>& member_ops);

// Drops any QKV candidate overlapping an op index already in `claimed_ops`
// (claimed by a higher-precedence pass), rebuilds the index maps for the
// retained set, and adds the retained candidates' op indices to
// `claimed_ops`.
void retain_unclaimed_qkv_fusions(
    std::vector<QkvBk64Fusion>& fusions,
    std::unordered_map<unsigned, size_t>& first_ops,
    std::unordered_map<unsigned, size_t>& last_ops,
    std::unordered_map<unsigned, size_t>& member_ops,
    std::unordered_set<unsigned>& claimed_ops);

// Emits the single fused q4gsw_qkv_bk64 dispatch for a matched pattern.
void add_qkv_bk64_dispatch(WebGPUGraph& graph, QkvBk64Fusion& fusion);

// Registers the dynamic-resize hook that switches the fusion between its
// fused and separate-projection dispatches as the live M crosses the BK64
// kernel's supported shapes.
void add_qkv_bk64_resize_hook(WebGPUGraph& graph, const QkvBk64Fusion& fusion);

} // namespace executorch::backends::webgpu::passes
