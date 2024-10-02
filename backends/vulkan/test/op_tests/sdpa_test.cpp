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

#include <cassert>
#include <iostream>

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
    const std::optional<at::Tensor> attn_mask,
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

vkcompute::vkapi::ScalarType from_at_scalartype(c10::ScalarType at_scalartype) {
  using namespace vkcompute;
  switch (at_scalartype) {
    case c10::kFloat:
      return vkapi::kFloat;
    case c10::kHalf:
      return vkapi::kHalf;
    case c10::kInt:
      return vkapi::kInt;
    case c10::kLong:
      return vkapi::kInt;
    case c10::kChar:
      return vkapi::kChar;
    default:
      VK_THROW("Unsupported at::ScalarType!");
  }
}

void test_vulkan_sdpa(
    const int start_input_pos,
    const int base_sequence_len,
    const int embedding_dim,
    const int num_heads,
    const int num_kv_heads,
    const int batch_size,
    const int max_seq_len,
    at::ScalarType dtype = at::kFloat) {
  const int head_dim = embedding_dim / num_heads;

  // K and V caches

  at::Tensor k_cache = at::zeros(
      {batch_size, max_seq_len, num_kv_heads, head_dim},
      at::device(at::kCPU).dtype(dtype));

  at::Tensor v_cache = at::zeros_like(k_cache);

  // Reference input data
  at::Tensor q = at::empty(
      {batch_size, max_seq_len, num_heads, head_dim},
      at::device(at::kCPU).dtype(dtype));
  at::Tensor k = at::empty_like(k_cache);
  at::Tensor v = at::empty_like(k_cache);

  // Get reference output
  at::Tensor out = at::empty_like(q);

  // Build Vulkan SDPA graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
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
      x.sizes().vec(), from_at_scalartype(x.scalar_type()));

  MAKE_INPUT_FOR(q);
  MAKE_INPUT_FOR(k);
  MAKE_INPUT_FOR(v);
#undef MAKE_INPUT_FOR

  const ValueRef r_input_pos_symint = graph.add_symint(start_input_pos);
  const ValueRef r_out = graph.add_tensor(
      out.sizes().vec(), from_at_scalartype(out.scalar_type()));

  VK_GET_OP_FN("llama.sdpa_with_kv_cache")
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

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  //
  // Run model
  //

#define COPY_INPUT(x) \
  graph.copy_into_staging(r_##x.staging, x.const_data_ptr(), x.numel());

#define EXTRACT_TENSOR(x)                             \
  at::Tensor vk_##x = at::zeros_like(x).contiguous(); \
  graph.copy_from_staging(                            \
      staging_##x, vk_##x.mutable_data_ptr(), vk_##x.numel());

  int seq_len = base_sequence_len;
  for (int i = 0, input_pos = start_input_pos;
       input_pos + seq_len < max_seq_len;
       input_pos += seq_len, i++) {
    q = at::rand(
        {batch_size, seq_len, num_heads, head_dim},
        at::device(at::kCPU).dtype(dtype));
    k = at::rand(
        {batch_size, seq_len, num_kv_heads, head_dim},
        at::device(at::kCPU).dtype(dtype));
    v = at::rand_like(k);

    at::Tensor reference_out = torch::executor::native::sdpa_with_kv_cache_aten(
        q, k, v, k_cache, v_cache, input_pos, seq_len, {}, 0.0, true, {});

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

    out = at::empty_like(q);
    EXTRACT_TENSOR(out);

    const bool output_correct = at::allclose(reference_out, vk_out);
    if (!output_correct) {
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

    // Vary sequence length between iterations
    seq_len = base_sequence_len + (i % 3);
  }
}

TEST(VulkanSDPATest, test_sdpa_op_small_params) {
  // Test SDPA with LLaMA3 model parameters
  const int starting_input_pos = 0;
  const int base_sequence_len = 3;
  const int embedding_dim = 18;
  const int num_heads = 6;
  const int num_kv_heads = 2;
  const int batch_size = 1;
  const int max_seq_len = 12;

  test_vulkan_sdpa(
      starting_input_pos,
      base_sequence_len,
      embedding_dim,
      num_heads,
      num_kv_heads,
      batch_size,
      max_seq_len);
}

TEST(VulkanSDPATest, test_sdpa_op_llama3_params) {
  // Test SDPA with LLaMA3 model parameters
  const int starting_input_pos = 0;
  const int base_sequence_len = 3;
  const int embedding_dim = 2048;
  const int num_heads = 32;
  const int num_kv_heads = 8;
  const int batch_size = 1;
  const int max_seq_len = 128;

  test_vulkan_sdpa(
      starting_input_pos,
      base_sequence_len,
      embedding_dim,
      num_heads,
      num_kv_heads,
      batch_size,
      max_seq_len);
}
