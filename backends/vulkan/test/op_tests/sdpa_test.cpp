/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#ifndef VULKAN_GENERAL_SDPA_ONLY
#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#endif

#include "test_utils.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>

#ifndef VULKAN_GENERAL_SDPA_ONLY

//
// SDPA Mode Enum
//

enum class SDPAMode { DECOMPOSED, FUSED, ATTN_WEIGHT_ONLY };

std::ostream& operator<<(std::ostream& os, const SDPAMode& mode) {
  switch (mode) {
    case SDPAMode::DECOMPOSED:
      return os << "DECOMPOSED";
    case SDPAMode::FUSED:
      return os << "FUSED";
    case SDPAMode::ATTN_WEIGHT_ONLY:
      return os << "ATTN_WEIGHT_ONLY";
  }
  return os;
}

namespace torch {
namespace executor {
namespace native {

// The below are copied from executorch/extension/llm/custom_ops/op_sdpa_aot.cpp
// They are needed because the original definitions are inaccessible due to
// being defined in an anonymous namespace.

Tensor& sdpa_with_kv_cache_out_no_context(
    const Tensor& q_projected,
    const Tensor& k_projected,
    const Tensor& v_projected,
    Tensor& key_cache,
    Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::native::sdpa_with_kv_cache_out(
      context,
      q_projected,
      k_projected,
      v_projected,
      key_cache,
      value_cache,
      start_pos,
      seq_len,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      output);
}

at::Tensor sdpa_with_kv_cache_aten(
    const at::Tensor& q_projected,
    const at::Tensor& k_projected,
    const at::Tensor& v_projected,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<at::Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale) {
  auto output = at::empty_like(q_projected);
  WRAP_TO_ATEN(sdpa_with_kv_cache_out_no_context, 11)
  (q_projected,
   k_projected,
   v_projected,
   key_cache,
   value_cache,
   start_pos,
   seq_len,
   attn_mask,
   dropout_p,
   is_causal,
   scale,
   output);
  return output;
}

} // namespace native
} // namespace executor
} // namespace torch

//
// Reference Implementation
//

/*
 * Converts a boolean mask to an additive mask. Values that are false are
 * converted to -inf, and values that are true are converted to 0.
 */
at::Tensor convert_boolean_attn_mask(
    const at::Tensor& attn_mask,
    caffe2::TypeMeta dtype) {
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask.dtype() == at::kBool) {
    return at::where(
        attn_mask.logical_not(),
        -std::numeric_limits<double>::infinity(),
        at::scalar_tensor(
            0.0, at::TensorOptions().dtype(dtype).device(attn_mask.device())));
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

/*
 * Construct an attention mask for SDPA.
 * 1. Construct a square matrix of ones with each dim equal to start_pos +
 *    seq_len
 * 2. Keep the lower triangular elements as 1 and set the rest to 0
 * 3. Slice the mask to keep only seq_len rows starting from input_pos
 * 4. Convert the mask to an additive mask
 */
at::Tensor construct_attention_mask(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const int start_pos) {
  const int max_seq_len = k_cache.size(1);
  const int seq_len = q.size(1);

  const int length = start_pos + seq_len;
  at::Tensor attn_mask_base =
      at::ones({length, length}, q.options().dtype(at::kBool)).tril();

  at::Tensor attn_mask_sliced =
      at::slice(attn_mask_base, 0, start_pos, start_pos + seq_len);

  attn_mask_sliced = convert_boolean_attn_mask(attn_mask_sliced, q.dtype());
  return attn_mask_sliced;
}

/*
 * Reference implementation of SDPA
 */
at::Tensor sdpa_reference_impl(
    const at::Tensor& q_projected,
    const at::Tensor& k_projected,
    const at::Tensor& v_projected,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    const std::optional<at::Tensor>& __attn_mask_ignored,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    SDPAMode mode = SDPAMode::DECOMPOSED) {
  at::Tensor attn_mask =
      construct_attention_mask(q_projected, key_cache, start_pos);

  // Cache update
  at::Tensor key_cache_updated = at::slice_scatter(
      key_cache, k_projected, 1, start_pos, start_pos + k_projected.size(1));
  at::Tensor value_cache_updated = at::slice_scatter(
      value_cache, v_projected, 1, start_pos, start_pos + v_projected.size(1));

  // Write back to input
  key_cache = key_cache_updated;
  value_cache = value_cache_updated;

  at::Tensor key_cache_sliced =
      at::slice(key_cache_updated, 1, 0, start_pos + q_projected.size(1));

  at::Tensor value_cache_sliced =
      at::slice(value_cache_updated, 1, 0, start_pos + q_projected.size(1));

  // Since n_heads may not be the same as n_kv_heads, the sliced k and v cache
  // matrices need to be "expanded" to match
  const int num_repeats = q_projected.size(2) / key_cache.size(2);
  at::Tensor key_cache_sliced_repeated =
      at::repeat_interleave(key_cache_sliced, num_repeats, 2);
  at::Tensor value_cache_sliced_repeated =
      at::repeat_interleave(value_cache_sliced, num_repeats, 2);

  at::Tensor q_transposed = q_projected.transpose(1, 2);
  at::Tensor k_transposed = key_cache_sliced_repeated.transpose(1, 2);
  at::Tensor v_transposed = value_cache_sliced_repeated.transpose(1, 2);

  at::Tensor k_transposed_2 = k_transposed.transpose(-2, -1);
  at::Tensor attn_weight_prescale = at::matmul(q_transposed, k_transposed_2);

  float scale_factor = 1.0 / sqrt(q_transposed.size(-1));
  at::Tensor attn_weight = attn_weight_prescale * scale_factor + attn_mask;

  if (mode == SDPAMode::ATTN_WEIGHT_ONLY) {
    return attn_weight;
  }

  at::Tensor attn_weight_softmax = at::softmax(attn_weight, -1);
  at::Tensor out = at::matmul(attn_weight_softmax, v_transposed);

  return out.transpose(1, 2);
}

//
// Test functions
//

void test_reference_sdpa(
    const int start_input_pos,
    const int sequence_len,
    const int head_dim,
    const int num_heads,
    const int num_kv_heads,
    const int batch_size,
    const int max_seq_len,
    at::ScalarType dtype = at::kFloat) {
  // K and V caches. Need an extra set for the reference implementation
  at::Tensor k_cache = at::zeros(
      {batch_size, max_seq_len, num_kv_heads, head_dim},
      at::device(at::kCPU).dtype(dtype));
  at::Tensor v_cache = at::zeros_like(k_cache);

  at::Tensor k_cache_ref = at::zeros_like(k_cache);
  at::Tensor v_cache_ref = at::zeros_like(v_cache);

  for (int input_pos = start_input_pos; input_pos + sequence_len < max_seq_len;
       input_pos += sequence_len) {
    at::Tensor q = at::rand(
        {batch_size, sequence_len, num_heads, head_dim},
        at::device(at::kCPU).dtype(dtype));
    at::Tensor k = at::rand(
        {batch_size, sequence_len, num_kv_heads, head_dim},
        at::device(at::kCPU).dtype(dtype));
    at::Tensor v = at::rand_like(k);

    at::Tensor reference_impl_out = sdpa_reference_impl(
        q, k, v, k_cache, v_cache, input_pos, sequence_len, {}, 0.0, true, {});

    at::Tensor reference_out = torch::executor::native::sdpa_with_kv_cache_aten(
        q,
        k,
        v,
        k_cache_ref,
        v_cache_ref,
        input_pos,
        sequence_len,
        {},
        0.0,
        true,
        {});

    ASSERT_TRUE(at::allclose(reference_impl_out, reference_out));
  }
}

void test_vulkan_sdpa(
    const int start_input_pos,
    const std::vector<int>& sequence_lens,
    const int head_dim,
    const int num_heads,
    const int num_kv_heads,
    const int batch_size,
    vkcompute::utils::StorageType storage_type,
    at::ScalarType dtype = at::kFloat,
    SDPAMode mode = SDPAMode::DECOMPOSED) {
  // compute the max sequence length
  int max_seq_len = start_input_pos;
  for (int i = 0; i < sequence_lens.size(); ++i) {
    max_seq_len += sequence_lens[i];
  }
  // Add some extra space to the max sequence length
  max_seq_len += 128;

  const int init_seq_len = max_seq_len;
  // K and V caches
  at::Tensor k_cache = at::zeros(
      {batch_size, max_seq_len, num_kv_heads, head_dim},
      at::device(at::kCPU).dtype(dtype));

  at::Tensor v_cache = at::zeros_like(k_cache);

  // Reference input data
  at::Tensor q = at::empty(
      {batch_size, init_seq_len, num_heads, head_dim},
      at::device(at::kCPU).dtype(dtype));
  at::Tensor k = at::empty(
      {batch_size, init_seq_len, num_kv_heads, head_dim},
      at::device(at::kCPU).dtype(dtype));
  at::Tensor v = at::empty_like(k);

  // Get reference output
  at::Tensor out = at::empty_like(q);
  if (mode == SDPAMode::ATTN_WEIGHT_ONLY) {
    out = at::empty({batch_size, num_heads, init_seq_len, init_seq_len});
  }

  // Build Vulkan SDPA graph
  using namespace vkcompute;

  GraphConfig config;
  ComputeGraph graph(config);

  // "Data" variant for vulkan initialization

  at::Tensor k_cache_data = at::zeros_like(k_cache);
  at::Tensor v_cache_data = at::zeros_like(v_cache);

#define MAKE_TENSORREF_FOR(x)              \
  ValueRef r_##x = graph.add_tensorref(    \
      x.sizes().vec(),                     \
      from_at_scalartype(x.scalar_type()), \
      x.const_data_ptr());

  MAKE_TENSORREF_FOR(k_cache_data);
  MAKE_TENSORREF_FOR(v_cache_data);

#define MAKE_INPUT_FOR(x)                    \
  IOValueRef r_##x = graph.add_input_tensor( \
      x.sizes().vec(), from_at_scalartype(x.scalar_type()), storage_type);

  MAKE_INPUT_FOR(q);
  MAKE_INPUT_FOR(k);
  MAKE_INPUT_FOR(v);
#undef MAKE_INPUT_FOR

  const ValueRef r_input_pos_symint = graph.add_symint(start_input_pos);
  const ValueRef r_out = graph.add_tensor(
      out.sizes().vec(), from_at_scalartype(out.scalar_type()), storage_type);

  switch (mode) {
    case SDPAMode::DECOMPOSED: {
      const ValueRef r_k_cache = graph.add_tensor(
          k_cache_data.sizes().vec(),
          from_at_scalartype(k_cache_data.scalar_type()),
          storage_type);
      const ValueRef r_v_cache = graph.add_tensor(
          v_cache_data.sizes().vec(),
          from_at_scalartype(v_cache_data.scalar_type()),
          storage_type);
      const ValueRef r_dummy_out = graph.add_tensor(
          {1}, from_at_scalartype(out.scalar_type()), utils::kBuffer);
      VK_GET_OP_FN("update_cache.default")
      (graph,
       {
           r_k.value,
           r_k_cache,
           r_input_pos_symint,
           r_dummy_out,
       });
      VK_GET_OP_FN("update_cache.default")
      (graph,
       {
           r_v.value,
           r_v_cache,
           r_input_pos_symint,
           r_dummy_out,
       });
      VK_GET_OP_FN("llama.custom_sdpa.default")
      (graph,
       {
           r_q.value,
           r_k_cache,
           r_v_cache,
           r_input_pos_symint,
           kDummyValueRef, // attn_mask
           kDummyValueRef, // dropout_p
           kDummyValueRef, // is_causal
           kDummyValueRef, // scale
           r_out,
       });
    } break;
    case SDPAMode::FUSED:
      VK_GET_OP_FN("sdpa_with_kv_cache.default")
      (graph,
       {
           r_q.value,
           r_k.value,
           r_v.value,
           r_k_cache_data,
           r_v_cache_data,
           r_input_pos_symint,
           kDummyValueRef, // sequence_len
           kDummyValueRef, // attn_mask
           kDummyValueRef, // dropout_p
           kDummyValueRef, // is_causal
           kDummyValueRef, // scale
           r_out,
       });
      break;
    case SDPAMode::ATTN_WEIGHT_ONLY:
      VK_GET_OP_FN("testing.compute_attn_weight_with_kv_cache.default")
      (graph,
       {
           r_q.value,
           r_k.value,
           r_v.value,
           r_k_cache_data,
           r_v_cache_data,
           r_input_pos_symint,
           kDummyValueRef, // sequence_len
           kDummyValueRef, // attn_mask
           kDummyValueRef, // dropout_p
           kDummyValueRef, // is_causal
           kDummyValueRef, // scale
           r_out,
       });
      break;
    default:
      VK_THROW("Unsupported SDPA mode");
  }

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();

  graph.prepack();

  //
  // Run model
  //

#define COPY_INPUT(x)                     \
  graph.maybe_cast_and_copy_into_staging( \
      r_##x.staging,                      \
      x.const_data_ptr(),                 \
      x.numel(),                          \
      from_at_scalartype(x.scalar_type()));

#define EXTRACT_TENSOR(x)                             \
  at::Tensor vk_##x = at::zeros_like(x).contiguous(); \
  graph.maybe_cast_and_copy_from_staging(             \
      staging_##x,                                    \
      vk_##x.mutable_data_ptr(),                      \
      vk_##x.numel(),                                 \
      from_at_scalartype(vk_##x.scalar_type()));

  torch::manual_seed(0);

  int input_pos = start_input_pos;
  for (auto seq_len : sequence_lens) {
    q = at::rand(
        {batch_size, seq_len, num_heads, head_dim},
        at::device(at::kCPU).dtype(dtype));
    k = at::rand(
        {batch_size, seq_len, num_kv_heads, head_dim},
        at::device(at::kCPU).dtype(dtype));
    v = at::rand_like(k);

    at::Tensor reference_out = sdpa_reference_impl(
        q, k, v, k_cache, v_cache, input_pos, seq_len, {}, 0.0, true, {}, mode);

    graph.set_symint(r_input_pos_symint, input_pos);
    graph.resize_input(0, q.sizes().vec());
    graph.resize_input(1, k.sizes().vec());
    graph.resize_input(2, v.sizes().vec());
    graph.propagate_resize();

    // Run Vulkan SDPA
    COPY_INPUT(q);
    COPY_INPUT(k);
    COPY_INPUT(v);

    graph.execute();

    if (mode == SDPAMode::ATTN_WEIGHT_ONLY) {
      const int context_len = input_pos + seq_len;
      const int context_len_align_up4 = (context_len + 3) & ~3;
      const int seq_len_align_up4 = (seq_len + 3) & ~3;

      out = at::empty(
          {batch_size, num_heads, seq_len_align_up4, context_len_align_up4},
          q.options());
    } else {
      out = at::empty_like(q);
    }
    EXTRACT_TENSOR(out);

    if (mode == SDPAMode::ATTN_WEIGHT_ONLY) {
      // Index vk_out to only include the relevant seq_len and context_len
      // dimensions
      int context_len = input_pos + seq_len;
      vk_out = vk_out.index(
          {at::indexing::Slice(),
           at::indexing::Slice(),
           at::indexing::Slice(0, seq_len),
           at::indexing::Slice(0, context_len)});
    }

    const bool output_correct = at::allclose(reference_out, vk_out);
    if (!output_correct) {
      // Print only differing tensor elements side by side for easier comparison
      auto ref_flat = reference_out.flatten();
      auto vk_flat = vk_out.flatten();
      auto numel = ref_flat.numel();
      std::cout << "While testing " << mode << " mode with " << storage_type
                << " storage" << std::endl;
      std::cout << "reference_out\tvk_out\tindex" << std::endl;
      int first_diff_idx = -1;
      auto sizes = reference_out.sizes();
      int d0 = sizes[0], d1 = sizes[1], d2 = sizes[2], d3 = sizes[3];
      for (int i = 0; i < numel; ++i) {
        if (std::abs(ref_flat[i].item<double>() - vk_flat[i].item<double>()) >
            1e-4) {
          // Compute 4-D index from flat index
          int i0 = i / (d1 * d2 * d3);
          int rem0 = i % (d1 * d2 * d3);
          int i1 = rem0 / (d2 * d3);
          int rem1 = rem0 % (d2 * d3);
          int i2 = rem1 / d3;
          int i3 = rem1 % d3;
          std::cout << ref_flat[i].item() << "\t" << vk_flat[i].item() << "\t["
                    << i0 << ", " << i1 << ", " << i2 << ", " << i3 << "]"
                    << std::endl;
          if (first_diff_idx == -1) {
            first_diff_idx = i;
          }
          break;
        }
      }
      if (first_diff_idx != -1) {
        // Compute 4-D index from flat index
        int i0 = first_diff_idx / (d1 * d2 * d3);
        int rem0 = first_diff_idx % (d1 * d2 * d3);
        int i1 = rem0 / (d2 * d3);
        int rem1 = rem0 % (d2 * d3);
        int i2 = rem1 / d3;
        int i3 = rem1 % d3;
        std::cout << "First difference at flat index " << first_diff_idx
                  << " which is tensor index [" << i0 << ", " << i1 << ", "
                  << i2 << ", " << i3 << "]" << std::endl;
      }

      at::Tensor diffs = at::abs(reference_out - vk_out);

      std::cout << "Failed at input_pos " << input_pos << " with seq_len "
                << seq_len << std::endl;

      std::cout << "Maximum difference: " << std::endl;
      std::cout << at::max(diffs).item() << std::endl;
      std::cout << "Found at index " << std::endl;
      std::cout << at::argmax(diffs).item() << std::endl;

      std::cout << "Maximum value observed: " << std::endl;
      std::cout << at::max(at::abs(at::cat({reference_out, vk_out}, -1))).item()
                << std::endl;
    }
    ASSERT_TRUE(output_correct);

    input_pos += seq_len;
  }
}

void test_vulkan_sdpa(
    const int start_input_pos,
    const std::vector<int>& sequence_lens,
    const int head_dim,
    const int num_heads,
    const int num_kv_heads,
    const int batch_size,
    at::ScalarType dtype = at::kFloat) {
  for (SDPAMode mode :
       {SDPAMode::ATTN_WEIGHT_ONLY, SDPAMode::DECOMPOSED, SDPAMode::FUSED}) {
    // Test texture
    test_vulkan_sdpa(
        start_input_pos,
        sequence_lens,
        head_dim,
        num_heads,
        num_kv_heads,
        batch_size,
        vkcompute::utils::kTexture3D,
        dtype,
        mode);

    // Test buffer
    test_vulkan_sdpa(
        start_input_pos,
        sequence_lens,
        head_dim,
        num_heads,
        num_kv_heads,
        batch_size,
        vkcompute::utils::kBuffer,
        dtype,
        mode);
  }
}

#endif // VULKAN_GENERAL_SDPA_ONLY

//
// General-purpose fused SDPA tests (et_vk.sdpa)
//

/*
 * Reference implementation of general SDPA: softmax(Q @ K^T * scale + bias) @ V
 * Q: [B, H, S, D], K: [B, H, L, D], V: [B, H, L, D]
 * Returns: [B, H, S, D]
 */
at::Tensor general_sdpa_reference_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& attn_mask = std::nullopt,
    const std::optional<double> scale = std::nullopt) {
  float scale_val =
      scale.has_value() ? scale.value() : (1.0 / sqrt(q.size(-1)));
  at::Tensor expanded_k = k;
  at::Tensor expanded_v = v;
  if (q.size(-3) != k.size(-3)) {
    const int64_t repeats = q.size(-3) / k.size(-3);
    expanded_k = at::repeat_interleave(k, repeats, -3);
    expanded_v = at::repeat_interleave(v, repeats, -3);
  }
  at::Tensor attn = at::matmul(q, expanded_k.transpose(-2, -1)) * scale_val;
  if (attn_mask.has_value()) {
    attn = attn + attn_mask.value();
  }
  attn = at::softmax(attn, -1);
  return at::matmul(attn, expanded_v);
}

at::Tensor ring_sdpa_reference_impl(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const int64_t start_pos,
    const int64_t window_size) {
  const int64_t seq_len = q.size(1);
  const int64_t cache_len = k_cache.size(1);
  const int64_t oldest_pos = std::max(start_pos - window_size + 1, 0L);
  const int64_t context_len = start_pos + seq_len - oldest_pos;
  at::Tensor cache_indices = at::remainder(
      at::arange(context_len, at::TensorOptions().dtype(at::kLong)) +
          oldest_pos,
      cache_len);

  at::Tensor k = k_cache.index_select(1, cache_indices);
  at::Tensor v = v_cache.index_select(1, cache_indices);
  if (q.size(2) != k.size(2)) {
    const int64_t repeats = q.size(2) / k.size(2);
    k = at::repeat_interleave(k, repeats, 2);
    v = at::repeat_interleave(v, repeats, 2);
  }

  const at::Tensor q_bhsd = q.transpose(1, 2).to(at::kFloat);
  const at::Tensor k_bhsd = k.transpose(1, 2).to(at::kFloat);
  const at::Tensor v_bhsd = v.transpose(1, 2).to(at::kFloat);
  at::Tensor attn = at::matmul(q_bhsd, k_bhsd.transpose(-2, -1)) /
      std::sqrt(static_cast<float>(q.size(-1)));

  at::Tensor mask = at::full(
      {seq_len, context_len},
      -std::numeric_limits<float>::infinity(),
      at::TensorOptions().dtype(at::kFloat));
  auto mask_access = mask.accessor<float, 2>();
  for (int64_t s = 0; s < seq_len; ++s) {
    const int64_t query_pos = start_pos + s;
    for (int64_t c = 0; c < context_len; ++c) {
      const int64_t key_pos = oldest_pos + c;
      const int64_t distance = query_pos - key_pos;
      if (distance >= 0 && distance < window_size) {
        mask_access[s][c] = 0.0f;
      }
    }
  }

  attn = at::softmax(attn + mask, -1);
  return at::matmul(attn, v_bhsd).transpose(1, 2);
}

void test_vulkan_ring_sdpa(
    const int64_t start_pos,
    const int64_t seq_len,
    const vkcompute::utils::StorageType storage_type,
    const at::ScalarType dtype = at::kFloat,
    const int64_t max_seq_len = -1,
    const bool buffer_cache = false,
    const int64_t window_size = 4) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 4;
  constexpr int64_t num_kv_heads = 2;
  constexpr int64_t head_dim = 8;
  const int64_t cache_len = window_size * 2;

  at::manual_seed(20260816 + start_pos + seq_len);
  at::Tensor q = at::rand(
      {batch_size, seq_len, num_heads, head_dim},
      at::TensorOptions().dtype(dtype));
  at::Tensor k_cache = at::zeros(
      {batch_size, cache_len, num_kv_heads, head_dim},
      at::TensorOptions().dtype(dtype));
  at::Tensor v_cache = at::zeros_like(k_cache);
  for (int64_t pos = 0; pos < start_pos + seq_len; ++pos) {
    k_cache.select(1, pos % cache_len)
        .copy_(at::rand({batch_size, num_kv_heads, head_dim}).to(dtype));
    v_cache.select(1, pos % cache_len)
        .copy_(at::rand({batch_size, num_kv_heads, head_dim}).to(dtype));
  }

  const at::Tensor reference_out =
      ring_sdpa_reference_impl(q, k_cache, v_cache, start_pos, window_size);

  using namespace vkcompute;
  GraphConfig config;
  ComputeGraph graph(config);
  std::vector<int64_t> graph_q_sizes = q.sizes().vec();
  if (max_seq_len > 0) {
    graph_q_sizes.at(1) = max_seq_len;
  }
  IOValueRef r_q = graph.add_input_tensor(
      graph_q_sizes, from_at_scalartype(dtype), storage_type);
  const utils::StorageType cache_storage =
      buffer_cache ? utils::kBuffer : storage_type;
  IOValueRef r_k = graph.add_input_tensor(
      k_cache.sizes().vec(), from_at_scalartype(dtype), cache_storage);
  IOValueRef r_v = graph.add_input_tensor(
      v_cache.sizes().vec(), from_at_scalartype(dtype), cache_storage);
  const ValueRef r_start_pos = graph.add_symint(start_pos);
  const ValueRef r_window_size = graph.add_scalar<int64_t>(window_size);
  const ValueRef r_out =
      graph.add_tensor(graph_q_sizes, from_at_scalartype(dtype), storage_type);

  VK_GET_OP_FN("et_vk.ring_sdpa.default")
  (graph, {r_q.value, r_k.value, r_v.value, r_start_pos, r_window_size, r_out});
  const ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();
  if (max_seq_len > 0) {
    graph.resize_input(0, q.sizes().vec());
    graph.propagate_resize();
  }
  graph.maybe_cast_and_copy_into_staging(
      r_q.staging, q.const_data_ptr(), q.numel(), from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_k.staging,
      k_cache.const_data_ptr(),
      k_cache.numel(),
      from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_v.staging,
      v_cache.const_data_ptr(),
      v_cache.numel(),
      from_at_scalartype(dtype));
  graph.execute();

  at::Tensor vk_out = at::zeros_like(q).contiguous();
  graph.maybe_cast_and_copy_from_staging(
      staging_out,
      vk_out.mutable_data_ptr(),
      vk_out.numel(),
      from_at_scalartype(dtype));

  const double tolerance = dtype == at::kHalf ? 1e-2 : 1e-4;
  const at::Tensor vk_out_float = vk_out.to(at::kFloat);
  if (!at::allclose(reference_out, vk_out_float, tolerance, tolerance)) {
    std::cout << "ring SDPA mismatch at start_pos=" << start_pos
              << ", seq_len=" << seq_len << ", max_seq_len=" << max_seq_len
              << ", storage=" << storage_type << std::endl;
    std::cout << "reference=" << reference_out.flatten() << std::endl;
    std::cout << "vulkan=" << vk_out_float.flatten() << std::endl;
    std::cout << "max_diff="
              << at::max(at::abs(reference_out - vk_out_float)).item<float>()
              << std::endl;
  }
  ASSERT_TRUE(at::allclose(reference_out, vk_out_float, tolerance, tolerance));
}

void test_vulkan_general_sdpa(
    const int batch_size,
    const int num_heads,
    const int q_seq_len,
    const int kv_seq_len,
    const int head_dim,
    const bool has_bias,
    at::ScalarType dtype = at::kFloat,
    const int requested_num_kv_heads = -1,
    const std::vector<int64_t>& requested_bias_shape = {},
    const bool causal_prefix_mask = false,
    const vkcompute::utils::StorageType storage_type =
        vkcompute::utils::kBuffer,
    const bool clone_kv_mask_to_storage = false,
    const bool permute_kv_before_clone = false,
    const bool update_kv_before_permute = false,
    const int max_q_seq_len = -1) {
  at::manual_seed(42);
  const int num_kv_heads =
      requested_num_kv_heads > 0 ? requested_num_kv_heads : num_heads;

  // Generate random inputs in [B, H, S, D] layout
  at::Tensor q = at::rand(
      {batch_size, num_heads, q_seq_len, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor k = at::rand(
      {batch_size, num_kv_heads, kv_seq_len, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor v = at::rand(
      {batch_size, num_kv_heads, kv_seq_len, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));

  std::optional<at::Tensor> bias = std::nullopt;
  if (has_bias) {
    const std::vector<int64_t> bias_shape = requested_bias_shape.empty()
        ? std::vector<int64_t>{batch_size, 1, 1, kv_seq_len}
        : requested_bias_shape;
    if (causal_prefix_mask) {
      bias = at::full(
          bias_shape,
          -std::numeric_limits<float>::infinity(),
          at::device(at::kCPU).dtype(at::kFloat));
      if (bias_shape.size() == 2) {
        auto mask = bias.value().accessor<float, 2>();
        for (int q_idx = 0; q_idx < q_seq_len; ++q_idx) {
          for (int k_idx = 0; k_idx <= q_idx; ++k_idx) {
            mask[q_idx][k_idx] = 0.0f;
          }
        }
      } else {
        auto mask = bias.value().accessor<float, 4>();
        for (int q_idx = 0; q_idx < q_seq_len; ++q_idx) {
          for (int k_idx = 0; k_idx <= q_idx; ++k_idx) {
            mask[0][0][q_idx][k_idx] = 0.0f;
          }
        }
      }
    } else {
      bias =
          at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat)) * 2.0 -
          1.0;
    }
  }

  // Compute reference output in fp32
  at::Tensor reference_out = general_sdpa_reference_impl(q, k, v, bias);

  // Cast to test dtype for Vulkan
  q = q.to(dtype);
  k = k.to(dtype);
  v = v.to(dtype);
  if (bias.has_value()) {
    bias = bias.value().to(dtype);
  }
  at::Tensor k_input = update_kv_before_permute
      ? k.slice(2, 0, q_seq_len).permute({0, 2, 1, 3}).contiguous()
      : (permute_kv_before_clone ? k.permute({0, 2, 1, 3}).contiguous() : k);
  at::Tensor v_input = update_kv_before_permute
      ? v.slice(2, 0, q_seq_len).permute({0, 2, 1, 3}).contiguous()
      : (permute_kv_before_clone ? v.permute({0, 2, 1, 3}).contiguous() : v);
  at::Tensor cache = at::zeros(
      {batch_size, kv_seq_len, num_kv_heads, head_dim},
      at::device(at::kCPU).dtype(dtype));
  at::Tensor indices =
      at::arange(q_seq_len, at::device(at::kCPU).dtype(at::kLong))
          .reshape({1, q_seq_len});

  // Build Vulkan compute graph
  using namespace vkcompute;

  GraphConfig config;
  ComputeGraph graph(config);
  std::vector<int64_t> graph_q_sizes = q.sizes().vec();
  if (max_q_seq_len > 0) {
    graph_q_sizes.at(2) = max_q_seq_len;
  }
  const utils::StorageType kv_input_storage =
      clone_kv_mask_to_storage ? utils::kBuffer : storage_type;
  const utils::StorageType projected_storage =
      update_kv_before_permute ? storage_type : kv_input_storage;

  IOValueRef r_q = graph.add_input_tensor(
      graph_q_sizes, from_at_scalartype(dtype), storage_type);
  IOValueRef r_k = graph.add_input_tensor(
      k_input.sizes().vec(), from_at_scalartype(dtype), projected_storage);
  IOValueRef r_v = graph.add_input_tensor(
      v_input.sizes().vec(), from_at_scalartype(dtype), projected_storage);

  ValueRef r_bias = kDummyValueRef;
  IOValueRef r_bias_io = {};
  if (has_bias) {
    std::vector<int64_t> graph_bias_sizes = bias.value().sizes().vec();
    if (max_q_seq_len > 0) {
      graph_bias_sizes.at(graph_bias_sizes.size() - 2) = max_q_seq_len;
    }
    r_bias_io = graph.add_input_tensor(
        graph_bias_sizes, from_at_scalartype(dtype), kv_input_storage);
    r_bias = r_bias_io.value;
  }

  ValueRef r_k_sdpa = r_k.value;
  ValueRef r_v_sdpa = r_v.value;
  ValueRef r_bias_sdpa = r_bias;
  IOValueRef r_k_cache_io = {};
  IOValueRef r_v_cache_io = {};
  IOValueRef r_indices_io = {};
  if (update_kv_before_permute) {
    r_k_cache_io = graph.add_input_tensor(
        cache.sizes().vec(), from_at_scalartype(dtype), utils::kBuffer);
    r_v_cache_io = graph.add_input_tensor(
        cache.sizes().vec(), from_at_scalartype(dtype), utils::kBuffer);
    r_indices_io = graph.add_input_tensor(
        indices.sizes().vec(), vkapi::kInt, utils::kBuffer);
    const ValueRef start_pos = graph.add_symint(0);
    const ValueRef k_dummy =
        graph.add_tensor({1}, from_at_scalartype(dtype), projected_storage);
    const ValueRef v_dummy =
        graph.add_tensor({1}, from_at_scalartype(dtype), projected_storage);
    VK_GET_OP_FN("update_cache_with_indices.default")
    (graph,
     {r_k.value, r_k_cache_io.value, start_pos, r_indices_io.value, k_dummy});
    VK_GET_OP_FN("update_cache_with_indices.default")
    (graph,
     {r_v.value, r_v_cache_io.value, start_pos, r_indices_io.value, v_dummy});
    r_k_sdpa = r_k_cache_io.value;
    r_v_sdpa = r_v_cache_io.value;
  }
  if (permute_kv_before_clone) {
    const ValueRef r_k_permuted = graph.add_tensor(
        k.sizes().vec(), from_at_scalartype(dtype), kv_input_storage);
    const ValueRef r_v_permuted = graph.add_tensor(
        v.sizes().vec(), from_at_scalartype(dtype), kv_input_storage);
    const ValueRef permute_dims = graph.add_scalar_list<int64_t>({0, 2, 1, 3});
    VK_GET_OP_FN("aten.permute_copy.default")
    (graph, {r_k_sdpa, permute_dims, r_k_permuted});
    VK_GET_OP_FN("aten.permute_copy.default")
    (graph, {r_v_sdpa, permute_dims, r_v_permuted});
    r_k_sdpa = r_k_permuted;
    r_v_sdpa = r_v_permuted;
  }
  if (clone_kv_mask_to_storage) {
    const ValueRef r_k_texture = graph.add_tensor(
        k.sizes().vec(), from_at_scalartype(dtype), storage_type);
    const ValueRef r_v_texture = graph.add_tensor(
        v.sizes().vec(), from_at_scalartype(dtype), storage_type);
    VK_GET_OP_FN("aten.clone.default")
    (graph, {r_k_sdpa, kDummyValueRef, r_k_texture});
    VK_GET_OP_FN("aten.clone.default")
    (graph, {r_v_sdpa, kDummyValueRef, r_v_texture});
    r_k_sdpa = r_k_texture;
    r_v_sdpa = r_v_texture;
    if (has_bias) {
      r_bias_sdpa = graph.add_tensor(
          bias.value().sizes().vec(), from_at_scalartype(dtype), storage_type);
      VK_GET_OP_FN("aten.clone.default")
      (graph, {r_bias, kDummyValueRef, r_bias_sdpa});
    }
  }

  const ValueRef r_out = graph.add_tensor(
      {batch_size,
       num_heads,
       max_q_seq_len > 0 ? max_q_seq_len : q_seq_len,
       head_dim},
      from_at_scalartype(dtype),
      storage_type);

  VK_GET_OP_FN("et_vk.sdpa.default")
  (graph,
   {
       r_q.value,
       r_k_sdpa,
       r_v_sdpa,
       r_bias_sdpa,
       kDummyValueRef, // scale (None -> 1/sqrt(head_dim))
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();
  if (max_q_seq_len > 0) {
    graph.resize_input(0, q.sizes().vec());
    if (has_bias) {
      graph.resize_input(3, bias.value().sizes().vec());
    }
    graph.propagate_resize();
  }

  // Copy inputs
  graph.maybe_cast_and_copy_into_staging(
      r_q.staging, q.const_data_ptr(), q.numel(), from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_k.staging,
      k_input.const_data_ptr(),
      k_input.numel(),
      from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_v.staging,
      v_input.const_data_ptr(),
      v_input.numel(),
      from_at_scalartype(dtype));
  if (update_kv_before_permute) {
    graph.maybe_cast_and_copy_into_staging(
        r_k_cache_io.staging,
        cache.const_data_ptr(),
        cache.numel(),
        from_at_scalartype(dtype));
    graph.maybe_cast_and_copy_into_staging(
        r_v_cache_io.staging,
        cache.const_data_ptr(),
        cache.numel(),
        from_at_scalartype(dtype));
    graph.maybe_cast_and_copy_into_staging(
        r_indices_io.staging,
        indices.const_data_ptr(),
        indices.numel(),
        vkapi::kLong);
  }
  if (has_bias) {
    graph.maybe_cast_and_copy_into_staging(
        r_bias_io.staging,
        bias.value().const_data_ptr(),
        bias.value().numel(),
        from_at_scalartype(dtype));
  }

  graph.execute();

  // Extract output
  at::Tensor vk_out = at::zeros(
                          {batch_size, num_heads, q_seq_len, head_dim},
                          at::device(at::kCPU).dtype(dtype))
                          .contiguous();
  graph.maybe_cast_and_copy_from_staging(
      staging_out,
      vk_out.mutable_data_ptr(),
      vk_out.numel(),
      from_at_scalartype(dtype));

  // Compare in fp32
  vk_out = vk_out.to(at::kFloat);

  // Use appropriate tolerance based on dtype
  double atol = dtype == at::kHalf ? 1e-2 : 1e-4;
  double rtol = dtype == at::kHalf ? 1e-2 : 1e-5;

  const bool output_correct = at::allclose(reference_out, vk_out, rtol, atol);
  if (causal_prefix_mask) {
    const at::Tensor diffs = at::abs(reference_out - vk_out);
    std::cout << "Causal prefix SDPA max diff: " << at::max(diffs).item()
              << ", mean diff: " << at::mean(diffs).item() << std::endl;
  }
  if (!output_correct) {
    at::Tensor diffs = at::abs(reference_out - vk_out);
    std::cout << "General SDPA test failed:" << " B=" << batch_size
              << " H=" << num_heads << " S=" << q_seq_len << " L=" << kv_seq_len
              << " D=" << head_dim << " bias=" << has_bias << " dtype=" << dtype
              << std::endl;
    std::cout << "Max diff: " << at::max(diffs).item() << std::endl;
    std::cout << "Max value: "
              << at::max(at::abs(at::cat({reference_out, vk_out}, -1))).item()
              << std::endl;

    // Print all elements for small tensors
    if (reference_out.numel() <= 64) {
      auto ref_flat = reference_out.flatten();
      auto vk_flat = vk_out.flatten();
      std::cout << "Reference vs Vulkan:" << std::endl;
      for (int i = 0; i < ref_flat.numel(); ++i) {
        std::cout << "  [" << i << "] ref=" << ref_flat[i].item<float>()
                  << " vk=" << vk_flat[i].item<float>() << " diff="
                  << std::abs(
                         ref_flat[i].item<float>() - vk_flat[i].item<float>())
                  << std::endl;
      }
    }
  }
  ASSERT_TRUE(output_correct);
}

// Basic correctness: small sizes, no bias, fp32
TEST(VulkanGeneralSDPATest, test_general_sdpa_small_no_bias) {
  test_vulkan_general_sdpa(1, 2, 4, 4, 8, false);
}

// With additive bias mask
TEST(VulkanGeneralSDPATest, test_general_sdpa_small_with_bias) {
  test_vulkan_general_sdpa(1, 2, 4, 8, 8, true);
}

// Cross-attention: Q and K have different sequence lengths
TEST(VulkanGeneralSDPATest, test_general_sdpa_cross_attention) {
  test_vulkan_general_sdpa(1, 4, 4, 16, 16, false);
}

// Batch size > 1
TEST(VulkanGeneralSDPATest, test_general_sdpa_batched) {
  test_vulkan_general_sdpa(2, 4, 8, 8, 16, false);
}

// Larger head_dim with bias (EdgeTAM-like)
TEST(VulkanGeneralSDPATest, test_general_sdpa_large_head_dim) {
  test_vulkan_general_sdpa(1, 8, 4, 4, 32, true);
}

// Non-aligned S (S is height dim, not width — no padding issue)
TEST(VulkanGeneralSDPATest, test_general_sdpa_non_aligned_s) {
  test_vulkan_general_sdpa(1, 2, 5, 4, 32, false);
}

// Large number of heads
TEST(VulkanGeneralSDPATest, test_general_sdpa_many_heads) {
  test_vulkan_general_sdpa(1, 8, 4, 8, 32, false);
}

// fp16 — validates fp32 internal accumulation
TEST(VulkanGeneralSDPATest, test_general_sdpa_fp16) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/4,
      /*q_seq_len=*/8,
      /*kv_seq_len=*/8,
      /*head_dim=*/16,
      /*has_bias=*/false,
      /*dtype=*/at::kHalf);
}

// fp16 with bias
TEST(VulkanGeneralSDPATest, test_general_sdpa_fp16_with_bias) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/4,
      /*q_seq_len=*/8,
      /*kv_seq_len=*/16,
      /*head_dim=*/16,
      /*has_bias=*/true,
      /*dtype=*/at::kHalf);
}

TEST(VulkanGeneralSDPATest, test_general_sdpa_gqa_broadcast_mask) {
  test_vulkan_general_sdpa(
      /*batch_size=*/2,
      /*num_heads=*/8,
      /*q_seq_len=*/4,
      /*kv_seq_len=*/16,
      /*head_dim=*/32,
      /*has_bias=*/true,
      /*dtype=*/at::kFloat,
      /*requested_num_kv_heads=*/2,
      /*requested_bias_shape=*/{1, 1, 4, 16});
}

TEST(VulkanGeneralSDPATest, test_general_sdpa_voxtral_encoder_shape) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/32,
      /*q_seq_len=*/4,
      /*kv_seq_len=*/128,
      /*head_dim=*/64,
      /*has_bias=*/true,
      /*dtype=*/at::kFloat,
      /*requested_num_kv_heads=*/32,
      /*requested_bias_shape=*/{4, 128});
}

TEST(VulkanGeneralSDPATest, test_general_sdpa_voxtral_gqa_fp16) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/32,
      /*q_seq_len=*/4,
      /*kv_seq_len=*/128,
      /*head_dim=*/128,
      /*has_bias=*/true,
      /*dtype=*/at::kHalf,
      /*requested_num_kv_heads=*/8,
      /*requested_bias_shape=*/{1, 1, 4, 128});
}

TEST(VulkanGeneralSDPATest, test_general_sdpa_voxtral_streaming_initial) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/32,
      /*q_seq_len=*/4,
      /*kv_seq_len=*/16384,
      /*head_dim=*/128,
      /*has_bias=*/true,
      /*dtype=*/at::kFloat,
      /*requested_num_kv_heads=*/8,
      /*requested_bias_shape=*/{1, 1, 4, 16384},
      /*causal_prefix_mask=*/true);
}

TEST(
    VulkanGeneralSDPATest,
    test_general_sdpa_voxtral_streaming_dynamic_buffer) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/32,
      /*q_seq_len=*/4,
      /*kv_seq_len=*/16384,
      /*head_dim=*/128,
      /*has_bias=*/true,
      /*dtype=*/at::kFloat,
      /*requested_num_kv_heads=*/8,
      /*requested_bias_shape=*/{4, 16384},
      /*causal_prefix_mask=*/true,
      /*storage_type=*/vkcompute::utils::kBuffer,
      /*clone_kv_mask_to_storage=*/false,
      /*permute_kv_before_clone=*/false,
      /*update_kv_before_permute=*/false,
      /*max_q_seq_len=*/2048);
}

TEST(
    VulkanGeneralSDPATest,
    test_general_sdpa_voxtral_streaming_initial_texture) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/32,
      /*q_seq_len=*/4,
      /*kv_seq_len=*/16384,
      /*head_dim=*/128,
      /*has_bias=*/true,
      /*dtype=*/at::kFloat,
      /*requested_num_kv_heads=*/8,
      /*requested_bias_shape=*/{4, 16384},
      /*causal_prefix_mask=*/true,
      /*storage_type=*/vkcompute::utils::kTexture3D);
}

TEST(
    VulkanGeneralSDPATest,
    test_general_sdpa_voxtral_streaming_initial_clones) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/32,
      /*q_seq_len=*/4,
      /*kv_seq_len=*/16384,
      /*head_dim=*/128,
      /*has_bias=*/true,
      /*dtype=*/at::kFloat,
      /*requested_num_kv_heads=*/8,
      /*requested_bias_shape=*/{4, 16384},
      /*causal_prefix_mask=*/true,
      /*storage_type=*/vkcompute::utils::kTexture3D,
      /*clone_kv_mask_to_storage=*/true);
}

TEST(
    VulkanGeneralSDPATest,
    test_general_sdpa_voxtral_streaming_initial_permute_clones) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/32,
      /*q_seq_len=*/4,
      /*kv_seq_len=*/16384,
      /*head_dim=*/128,
      /*has_bias=*/true,
      /*dtype=*/at::kFloat,
      /*requested_num_kv_heads=*/8,
      /*requested_bias_shape=*/{4, 16384},
      /*causal_prefix_mask=*/true,
      /*storage_type=*/vkcompute::utils::kTexture3D,
      /*clone_kv_mask_to_storage=*/true,
      /*permute_kv_before_clone=*/true);
}

TEST(
    VulkanGeneralSDPATest,
    test_general_sdpa_voxtral_streaming_initial_cache_chain) {
  test_vulkan_general_sdpa(
      /*batch_size=*/1,
      /*num_heads=*/32,
      /*q_seq_len=*/4,
      /*kv_seq_len=*/16384,
      /*head_dim=*/128,
      /*has_bias=*/true,
      /*dtype=*/at::kFloat,
      /*requested_num_kv_heads=*/8,
      /*requested_bias_shape=*/{4, 16384},
      /*causal_prefix_mask=*/true,
      /*storage_type=*/vkcompute::utils::kTexture3D,
      /*clone_kv_mask_to_storage=*/true,
      /*permute_kv_before_clone=*/true,
      /*update_kv_before_permute=*/true);
}

TEST(VulkanRingSDPATest, test_decoder_wrap_boundaries) {
  for (const auto storage_type :
       {vkcompute::utils::kBuffer, vkcompute::utils::kTexture3D}) {
    for (const int64_t start_pos : {0, 3, 4, 7, 8}) {
      test_vulkan_ring_sdpa(start_pos, 1, storage_type);
    }
  }
}

TEST(VulkanRingSDPATest, test_encoder_wrap_boundaries) {
  for (const auto storage_type :
       {vkcompute::utils::kBuffer, vkcompute::utils::kTexture3D}) {
    for (const int64_t start_pos : {0, 4, 8}) {
      test_vulkan_ring_sdpa(start_pos, 4, storage_type);
    }
  }
}

TEST(VulkanRingSDPATest, test_encoder_wrap_fp16) {
  test_vulkan_ring_sdpa(8, 4, vkcompute::utils::kBuffer, /*dtype=*/at::kHalf);
}

TEST(VulkanRingSDPATest, test_texture_io_buffer_cache) {
  test_vulkan_ring_sdpa(
      8,
      1,
      vkcompute::utils::kTexture3D,
      /*dtype=*/at::kFloat,
      /*max_seq_len=*/-1,
      /*buffer_cache=*/true);
  test_vulkan_ring_sdpa(
      8,
      4,
      vkcompute::utils::kTexture3D,
      /*dtype=*/at::kHalf,
      /*max_seq_len=*/-1,
      /*buffer_cache=*/true);
}

TEST(VulkanRingSDPATest, test_two_chunk_encoder) {
  for (const int64_t start_pos : {0, 8, 12, 16}) {
    test_vulkan_ring_sdpa(
        start_pos,
        8,
        vkcompute::utils::kTexture3D,
        /*dtype=*/at::kFloat,
        /*max_seq_len=*/-1,
        /*buffer_cache=*/true,
        /*window_size=*/8);
  }
}

TEST(VulkanRingSDPATest, test_dynamic_decoder_uses_tiled_path) {
  for (const auto storage_type :
       {vkcompute::utils::kBuffer, vkcompute::utils::kTexture3D}) {
    test_vulkan_ring_sdpa(
        0,
        1,
        storage_type,
        /*dtype=*/at::kFloat,
        /*max_seq_len=*/4);
  }
}

TEST(VulkanRingSDPATest, test_rejects_resize_beyond_allocated_capacity) {
  using namespace vkcompute;

  GraphConfig config;
  ComputeGraph graph(config);
  IOValueRef q =
      graph.add_input_tensor({1, 4, 4, 8}, vkapi::kFloat, utils::kBuffer);
  IOValueRef k =
      graph.add_input_tensor({1, 8, 2, 8}, vkapi::kFloat, utils::kBuffer);
  IOValueRef v =
      graph.add_input_tensor({1, 8, 2, 8}, vkapi::kFloat, utils::kBuffer);
  const ValueRef start_pos = graph.add_symint(8);
  const ValueRef window_size = graph.add_scalar<int64_t>(4);
  const ValueRef out =
      graph.add_tensor({1, 4, 4, 8}, vkapi::kFloat, utils::kBuffer);

  VK_GET_OP_FN("et_vk.ring_sdpa.default")
  (graph, {q.value, k.value, v.value, start_pos, window_size, out});
  graph.set_output_tensor(out);
  graph.prepare();
  graph.prepack();

  graph.resize_input(0, {1, 5, 4, 8});
  EXPECT_THROW(graph.propagate_resize(), vkapi::Error);
}

#ifndef VULKAN_GENERAL_SDPA_ONLY

//
// Existing KV-cache SDPA tests
//

TEST(VulkanSDPATest, test_sdpa_op_small_params) {
  const int base_sequence_len = 3;
  const int num_heads = 8;
  const int head_dim = 4;
  const int num_kv_heads = 4;

  test_vulkan_sdpa(
      0, {3, 1, 1, 5, 1, 1, 2}, head_dim, num_heads, num_kv_heads, 1);
}

TEST(VulkanSDPATest, test_sdpa_op_small_params_dynamic) {
  const int base_sequence_len = 3;
  const int head_dim = 8;
  const int num_heads = 6;
  const int num_kv_heads = 2;

  test_vulkan_sdpa(0, {3, 1, 1, 5, 1, 1}, head_dim, num_heads, num_kv_heads, 1);
}

TEST(VulkanSDPATest, test_sdpa_op_llama3_params_dynamic) {
  const int head_dim = 128;
  const int num_heads = 24;
  const int num_kv_heads = 8;

  test_vulkan_sdpa(
      0, {111, 1, 1, 1, 57, 1, 1}, head_dim, num_heads, num_kv_heads, 1);
}

// GQA group size G = Hq / Hkv = 4 (Llama-style). The other decode tests cover
// G=2 (small_params: 8/4) and G=3 (small_params_dynamic: 6/2, llama3: 24/8);
// this exercises the G=4 path of the AV coop-GQA shader, which reuses each V
// texel across all query heads in a group.
TEST(VulkanSDPATest, test_sdpa_op_gqa_group4) {
  const int head_dim = 64;
  const int num_heads = 8;
  const int num_kv_heads = 2;

  test_vulkan_sdpa(
      0, {5, 1, 1, 1, 1, 3, 1}, head_dim, num_heads, num_kv_heads, 1);
}

#endif // VULKAN_GENERAL_SDPA_ONLY
