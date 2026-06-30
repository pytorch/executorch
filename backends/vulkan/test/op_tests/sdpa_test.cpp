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

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/extension/llm/custom_ops/op_sdpa.h>

#include "test_utils.h"

#include <cassert>
#include <iostream>

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
  at::Tensor attn = at::matmul(q, k.transpose(-2, -1)) * scale_val;
  if (attn_mask.has_value()) {
    attn = attn + attn_mask.value();
  }
  attn = at::softmax(attn, -1);
  return at::matmul(attn, v);
}

void test_vulkan_general_sdpa(
    const int batch_size,
    const int num_heads,
    const int q_seq_len,
    const int kv_seq_len,
    const int head_dim,
    const bool has_bias,
    at::ScalarType dtype = at::kFloat) {
  torch::manual_seed(42);

  // Generate random inputs in [B, H, S, D] layout
  at::Tensor q = at::rand(
      {batch_size, num_heads, q_seq_len, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor k = at::rand(
      {batch_size, num_heads, kv_seq_len, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor v = at::rand(
      {batch_size, num_heads, kv_seq_len, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));

  std::optional<at::Tensor> bias = std::nullopt;
  if (has_bias) {
    // Broadcastable bias: [B, 1, 1, kv_seq_len]
    bias = at::rand(
               {batch_size, 1, 1, kv_seq_len},
               at::device(at::kCPU).dtype(at::kFloat)) *
            2.0 -
        1.0;
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

  // Build Vulkan compute graph
  using namespace vkcompute;

  GraphConfig config;
  ComputeGraph graph(config);

  IOValueRef r_q = graph.add_input_tensor(
      q.sizes().vec(), from_at_scalartype(dtype), utils::kBuffer);
  IOValueRef r_k = graph.add_input_tensor(
      k.sizes().vec(), from_at_scalartype(dtype), utils::kBuffer);
  IOValueRef r_v = graph.add_input_tensor(
      v.sizes().vec(), from_at_scalartype(dtype), utils::kBuffer);

  ValueRef r_bias = kDummyValueRef;
  IOValueRef r_bias_io = {};
  if (has_bias) {
    r_bias_io = graph.add_input_tensor(
        bias.value().sizes().vec(), from_at_scalartype(dtype), utils::kBuffer);
    r_bias = r_bias_io.value;
  }

  const ValueRef r_out = graph.add_tensor(
      {batch_size, num_heads, q_seq_len, head_dim},
      from_at_scalartype(dtype),
      utils::kBuffer);

  VK_GET_OP_FN("et_vk.sdpa.default")
  (graph,
   {
       r_q.value,
       r_k.value,
       r_v.value,
       r_bias,
       kDummyValueRef, // scale (None -> 1/sqrt(head_dim))
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();

  // Copy inputs
  graph.maybe_cast_and_copy_into_staging(
      r_q.staging, q.const_data_ptr(), q.numel(), from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_k.staging, k.const_data_ptr(), k.numel(), from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_v.staging, v.const_data_ptr(), v.numel(), from_at_scalartype(dtype));
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
