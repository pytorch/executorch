/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/SDPA.h>

#include <cmath>

namespace vkcompute {

// Bare-minimum mirror of the LLM decode/prefill path in sdpa_impl() (see
// SDPA.cpp), stripped of the input-validation VK_CHECK_CONDs. Building the
// three SDPA nodes directly lets the test forward a shader_override knob to the
// AV node, which the production op boundary (llama.custom_sdpa) can't carry.
static void test_sdpa_impl(
    ComputeGraph& graph,
    const ValueRef q_projected,
    const ValueRef k_cache,
    const ValueRef v_cache,
    const ValueRef input_pos_symint,
    const ValueRef out,
    const ValueRef shader_override) {
  const int64_t num_q_heads = graph.size_at<int64_t>(-2, q_projected);
  int64_t max_seq_len = graph.size_at<int64_t>(-3, q_projected);
  const int64_t max_context_len = graph.size_at<int32_t>(-3, k_cache);

  const utils::StorageType attn_weights_storage =
      graph.storage_type_of(q_projected);

  // S and context dims are padded to a multiple of 4 to match sdpa_impl() (see
  // the allocation comment there): the buffer SDPA shaders index each head at
  // an align_up_4(S)/align_up_4(context) stride, so the allocation must reserve
  // the padded extent or later heads land out of bounds on the buffer path.
  const int64_t padded_context_len = utils::align_up_4(max_context_len);

  // Clamp attn-weight seq_len so the PADDED buffer stays within the numel limit
  // (mirrors sdpa_impl; a no-op for the texture path the harness uses). The
  // check must use the padded sizes actually allocated, not the raw ones.
  if (attn_weights_storage == utils::kBuffer) {
    const int64_t max_buffer_numel = graph.max_buffer_numel();
    if (num_q_heads * utils::align_up_4(max_seq_len) * padded_context_len >=
        max_buffer_numel) {
      max_seq_len = max_buffer_numel / (num_q_heads * padded_context_len);
      if (max_seq_len % 4 != 0) {
        max_seq_len = (max_seq_len / 4) * 4;
      } else {
        max_seq_len -= 4;
      }
    }
  }

  const std::vector<int64_t> attn_weight_full_sizes = {
      1, num_q_heads, utils::align_up_4(max_seq_len), padded_context_len};

  TmpTensor attn_weights(
      &graph,
      attn_weight_full_sizes,
      graph.dtype_of(q_projected),
      attn_weights_storage,
      utils::kWidthPacked);

  TmpTensor attn_weights_softmax(
      &graph,
      attn_weight_full_sizes,
      graph.dtype_of(q_projected),
      attn_weights_storage,
      utils::kWidthPacked);

  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  add_sdpa_compute_attn_weights_node(
      graph,
      q_projected,
      k_cache,
      input_pos_symint,
      /*attn_mask=*/kDummyValueRef,
      scale_val,
      attn_weights,
      SDPAMode::LLM);

  add_sdpa_attn_weights_softmax_node(
      graph,
      attn_weights,
      q_projected,
      k_cache,
      input_pos_symint,
      attn_weights_softmax,
      SDPAMode::LLM);

  add_sdpa_compute_out_node(
      graph,
      attn_weights_softmax,
      v_cache,
      q_projected,
      /*k=*/kDummyValueRef,
      input_pos_symint,
      out,
      SDPAMode::LLM,
      shader_override);
}

// Test wrapper for the LLM KV-cache SDPA op (llama.custom_sdpa.default).
//
// The production op reads the dynamic context length from an `input_pos`
// symint, but input_pos is not a free parameter: the op enforces
// context_len = S + input_pos, and context_len is exactly the KV-cache's dim
// -3. Since q is [B=1, S, H, D] and the caches are [B=1, context_len, H_kv, D],
// input_pos is fully determined by the tensor shapes. Deriving it here from
// those shapes keeps the single source of truth (the cache size) authoritative.
//
// Decode vs prefill is selected automatically inside the nodes via
// is_single_token() (S == 1 -> coop/GEMV shaders, S > 1 -> tiled shaders). For
// single-token decode the AV shader is picked by impl_selector:
//   "default"   -> auto-select (GQA-reuse coop shader when it applies)
//   "gqa"       -> force the GQA-reuse coop shader (vendor picks base vs tile2)
//   "gqa_tile2" -> force the head_dim output-tiled GQA variant (any device)
//   "gqa_base"  -> force the base (non-tiled) GQA variant (any device)
//   "non_gqa"   -> force the per-query-head coop shader
// impl_selector has no effect on prefill (tiled).
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
      impl_selector == "default" || impl_selector == "gqa" ||
          impl_selector == "gqa_tile2" || impl_selector == "gqa_base" ||
          impl_selector == "non_gqa",
      "test_sdpa: impl_selector must be one of {default, gqa, gqa_tile2, "
      "gqa_base, non_gqa}");

  const int64_t seq_len = graph.size_at<int64_t>(-3, q);
  const int64_t context_len = graph.size_at<int64_t>(-3, k_cache);
  VK_CHECK_COND(context_len >= seq_len);
  const int32_t input_pos_val =
      utils::safe_downcast<int32_t>(context_len - seq_len);

  const ValueRef input_pos_symint = graph.add_symint(input_pos_val);

  // shader_override (see SDPA.h): kDummyValueRef auto; otherwise a
  // kShaderOverride* scalar forcing the AV shader family / variant.
  ValueRef shader_override = kDummyValueRef;
  if (impl_selector == "gqa") {
    shader_override = graph.add_scalar<int64_t>(kShaderOverrideForceGqa);
  } else if (impl_selector == "gqa_tile2") {
    shader_override = graph.add_scalar<int64_t>(kShaderOverrideForceTile2);
  } else if (impl_selector == "gqa_base") {
    shader_override = graph.add_scalar<int64_t>(kShaderOverrideForceBase);
  } else if (impl_selector == "non_gqa") {
    shader_override = graph.add_scalar<int64_t>(kShaderOverrideForceNonGqa);
  }

  test_sdpa_impl(
      graph, q, k_cache, v_cache, input_pos_symint, out, shader_override);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_sdpa.default, test_sdpa);
}

} // namespace vkcompute
