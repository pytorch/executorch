/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

//
// SDPA mode: distinguishes the two dispatch families sharing SDPA.cpp.
//   LLM   — Llama-style KV-cache SDPA. Q layout [B=1, S, H,    D] (DHSB).
//           Separate k_cache/v_cache inputs + input_pos_symint for dynamic
//           context_len. attn_weights are padded to multiples of 4 in the
//           S/context_len dims and carry the input dtype. A coop (GEMV)
//           shader variant is selected for single-token decode.
//   FUSED — General SDPA fused op. Q layout [B, H, S, D] (DSHB). No cache,
//           optional additive attn_mask, optional scale arg. attn_weights
//           are unpadded and always fp32. Tiled shader variant only.
//
enum class SDPAMode { LLM, FUSED };

void add_sdpa_compute_attn_weights_node(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_mask,
    const float scale_val,
    const ValueRef attn_weights,
    const SDPAMode mode);

void add_sdpa_attn_weights_softmax_node(
    ComputeGraph& graph,
    const ValueRef attn_weights,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax,
    const SDPAMode mode);

// shader_override: test-only knob selecting the single-token decode AV shader.
//   kDummyValueRef (-1) — auto: use the GQA-reuse coop shader
//                         (`sdpa_compute_out_gqa_coop`) when it applies.
//   0                   — force the per-query-head coop shader
//                         (`sdpa_compute_out_coop`).
//   1                   — force the GQA-reuse coop shader.
// Lets the benchmark/test exercise both AV shaders on the same shape. Forcing
// GQA (1) requires a GQA-eligible shape (Hq divisible by Hkv, group size <= the
// shader's compile-time bound); this is VK_CHECK'd, so an ineligible forced
// shape fails loudly rather than silently dropping query heads.
void add_sdpa_compute_out_node(
    ComputeGraph& graph,
    const ValueRef attn_weights_softmax,
    const ValueRef v,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef out,
    const SDPAMode mode,
    const ValueRef shader_override = kDummyValueRef);

} // namespace vkcompute
