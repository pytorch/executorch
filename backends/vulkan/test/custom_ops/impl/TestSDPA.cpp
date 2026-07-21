/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

namespace vkcompute {

// Test wrapper for the LLM KV-cache SDPA op (llama.custom_sdpa.default).
//
// The production op reads the dynamic context length from an `input_pos`
// symint, but input_pos is not a free parameter: the op enforces
// context_len = S + input_pos, and context_len is exactly the KV-cache's dim
// -3. Since q is [B=1, S, H, D] and the caches are [B=1, context_len, H_kv, D],
// input_pos is fully determined by the tensor shapes. Deriving it here from
// those shapes keeps the single source of truth (the cache size) authoritative;
// exposing input_pos as its own ValueSpec would add a redundant degree of
// freedom that a test could set inconsistent with the cache shape.
//
// Decode vs prefill is selected automatically inside the op via
// is_single_token() (S == 1 -> coop/GEMV shaders, S > 1 -> tiled shaders), so
// impl_selector is accepted for interface uniformity but only "default" is
// meaningful here.
//
// Args: q, k_cache, v_cache, impl_selector, out
void test_sdpa(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q = args.at(arg_idx++);
  const ValueRef k_cache = args.at(arg_idx++);
  const ValueRef v_cache = args.at(arg_idx++);
  const ValueRef impl_selector_str = args.at(arg_idx++);
  const ValueRef out = args.at(arg_idx++);

  const std::string impl_selector = graph.extract_string(impl_selector_str);
  VK_CHECK_COND(
      impl_selector == "default",
      "test_sdpa only supports the 'default' impl_selector");

  const int64_t seq_len = graph.size_at<int64_t>(-3, q);
  const int64_t context_len = graph.size_at<int64_t>(-3, k_cache);
  VK_CHECK_COND(context_len >= seq_len);
  const int32_t input_pos_val =
      utils::safe_downcast<int32_t>(context_len - seq_len);

  const ValueRef input_pos_symint = graph.add_symint(input_pos_val);

  VK_GET_OP_FN("llama.custom_sdpa.default")
  (graph,
   {
       q,
       k_cache,
       v_cache,
       input_pos_symint,
       kDummyValueRef, // attn_mask
       kDummyValueRef, // dropout_p
       kDummyValueRef, // is_causal (implementation assumes causal)
       kDummyValueRef, // scale (implementation computes 1/sqrt(head_dim))
       out,
   });
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_sdpa.default, test_sdpa);
}

} // namespace vkcompute
