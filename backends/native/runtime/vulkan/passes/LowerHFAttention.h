// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn::vulkan {

inline constexpr char kRopeTableAttr[] = "vulkan_rope_table";
inline constexpr char kCachePositionAttr[] = "vulkan_cache_position";
inline constexpr char kWrittenBeforeReadAttr[] = "vulkan_written_before_read";
// Marks a stored constant the rewrite assumed is zero; the engine checks it.
inline constexpr char kZeroConstantAttr[] = "vulkan_zero_constant";

// A [max_seq_len, rotary_dim] cos or sin table the engine computes from a
// stored inv_freq constant: table[p][j] = f(p * inv_freq[j % (rotary_dim / 2)])
// * attention_scale, with f = cos or sin.
struct RopeTable {
  ValueId inv_freq_id;
  bool use_sin;
  double attention_scale;
};

// Marks a graph input the lowered kernels read as the positions
// cache_position[0] through cache_position[0] + S - 1 of caches holding
// `cache_len` rows.
struct CachePosition {
  int64_t cache_len;
};

// Throws std::runtime_error unless `positions` are contiguous, fit within
// `bound.cache_len` rows and start no later than `rows_written`, the rows
// earlier calls filled; then extends `rows_written`. The kernels would
// otherwise attend over the wrong or never-written rows, or write past the
// cache.
void check_cache_positions(
    std::span<const int64_t> positions,
    const CachePosition& bound,
    int64_t& rows_written);

// Rewrites the HF static-cache attention block
//   sdpa(rope(q), index_put_(k_cache, [None, None, cache_position], rope(k)),
//        index_put_(v_cache, [None, None, cache_position], v), causal_mask)
// into apply_rotary_emb_hf, llama.update_cache and causal llama.custom_sdpa,
// re-laying the zero-initialized caches out as [1, S, H_kv, D]. cache_position
// must be a graph input, and causal_mask HF's static-cache causal mask over it.
// The kernels read positions from cache_position[0] on, so it must hold
// contiguous positions; the pass marks it with kCachePositionAttr for the
// engine to check on the host. Since that check also keeps the kernels from
// reading rows no call has written, the pass marks the caches, which only the
// lowered kernels use, with kWrittenBeforeReadAttr: their zero initialization
// is unobservable. The kernels also assume the mask's kv_offset constant is
// zero, so the pass marks it with kZeroConstantAttr. Blocks that do not match
// are left unchanged.
//
// TODO(Native-VK): temporary. This matches HF's in-graph static cache and mask
// structurally, so any other attention form fails to load as an unsupported op.
// Remove it once the HF path keeps its KV cache off-graph.
size_t lower_hf_attention(Graph& graph);

} // namespace ptn::vulkan
