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

#include "test_utils.h"

#include <cassert>

//
// Reference Implementations
//

std::pair<at::Tensor, at::Tensor> rotary_embedding_impl(
    const at::Tensor& xq,
    const at::Tensor& xk,
    const at::Tensor& freqs_cos,
    const at::Tensor& freqs_sin) {
  std::vector<at::Tensor> xq_even_odd = at::unbind(
      xq.reshape({xq.size(0), xq.size(1), xq.size(2), xq.size(3) / 2, 2}), -1);
  at::Tensor& xq_r = xq_even_odd[0];
  at::Tensor& xq_i = xq_even_odd[1];

  std::vector<at::Tensor> xk_even_odd = at::unbind(
      xk.reshape({xk.size(0), xk.size(1), xk.size(2), xk.size(3) / 2, 2}), -1);
  at::Tensor& xk_r = xk_even_odd[0];
  at::Tensor& xk_i = xk_even_odd[1];

  at::Tensor freqs_cos_reshape =
      freqs_cos.reshape({1, freqs_cos.size(0), 1, freqs_cos.size(1)});
  at::Tensor freqs_sin_reshape =
      freqs_sin.reshape({1, freqs_sin.size(0), 1, freqs_sin.size(1)});

  at::Tensor xq_out_r = xq_r * freqs_cos_reshape - xq_i * freqs_sin_reshape;
  at::Tensor xq_out_i = xq_r * freqs_sin_reshape + xq_i * freqs_cos_reshape;
  at::Tensor xk_out_r = xk_r * freqs_cos_reshape - xk_i * freqs_sin_reshape;
  at::Tensor xk_out_i = xk_r * freqs_sin_reshape + xk_i * freqs_cos_reshape;

  at::Tensor xq_out = at::flatten(at::stack({xq_out_r, xq_out_i}, -1), 3);
  at::Tensor xk_out = at::flatten(at::stack({xk_out_r, xk_out_i}, -1), 3);

  return std::make_pair(xq_out, xk_out);
}

//
// Test functions
//

void test_reference(
    const int n_heads = 4,
    const int n_kv_heads = 2,
    const int dim = 32,
    const int seq_len = 1) {
  const int head_dim = dim / n_heads;

  at::Tensor xq = at::rand(
      {1, seq_len, n_heads, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor xk = at::rand(
      {1, seq_len, n_kv_heads, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor freqs_cos =
      at::rand({seq_len, head_dim / 2}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor freqs_sin =
      at::rand({seq_len, head_dim / 2}, at::device(at::kCPU).dtype(at::kFloat));

  std::pair<at::Tensor, at::Tensor> outs =
      rotary_embedding_impl(xq, xk, freqs_cos, freqs_sin);
  at::Tensor& xq_out = outs.first;
  at::Tensor& xk_out = outs.second;

  // Build Vulkan graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

#define MAKE_INPUT_FOR(x)                    \
  IOValueRef r_##x = graph.add_input_tensor( \
      x.sizes().vec(), from_at_scalartype(x.scalar_type()));

  MAKE_INPUT_FOR(xq);
  MAKE_INPUT_FOR(xk);
  MAKE_INPUT_FOR(freqs_cos);
  MAKE_INPUT_FOR(freqs_sin);

  const ValueRef r_xq_out = graph.add_tensor(
      xq_out.sizes().vec(), from_at_scalartype(xq_out.scalar_type()));
  const ValueRef r_xk_out = graph.add_tensor(
      xk_out.sizes().vec(), from_at_scalartype(xk_out.scalar_type()));

  VK_GET_OP_FN("et_vk.apply_rotary_emb.default")
  (graph,
   {r_xq.value,
    r_xk.value,
    r_freqs_cos.value,
    r_freqs_sin.value,
    graph.add_value_list({r_xq_out, r_xk_out})});

  ValueRef staging_xq_out = graph.set_output_tensor(r_xq_out);
  ValueRef staging_xk_out = graph.set_output_tensor(r_xk_out);

  graph.prepare();

  graph.prepack();

  //
  // Run model
  //

  graph.propagate_resize();
  graph.maybe_cast_and_copy_into_staging(
      r_xq.staging,
      xq.const_data_ptr(),
      xq.numel(),
      from_at_scalartype(xq.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_xk.staging,
      xk.const_data_ptr(),
      xk.numel(),
      from_at_scalartype(xk.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_freqs_cos.staging,
      freqs_cos.const_data_ptr(),
      freqs_cos.numel(),
      from_at_scalartype(freqs_cos.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_freqs_sin.staging,
      freqs_sin.const_data_ptr(),
      freqs_sin.numel(),
      from_at_scalartype(freqs_sin.scalar_type()));

  graph.execute();

  at::Tensor vk_xq_out = at::empty_like(xq_out);
  graph.maybe_cast_and_copy_from_staging(
      staging_xq_out,
      vk_xq_out.mutable_data_ptr(),
      vk_xq_out.numel(),
      from_at_scalartype(vk_xq_out.scalar_type()));

  at::Tensor vk_xk_out = at::empty_like(xk_out);
  graph.maybe_cast_and_copy_from_staging(
      staging_xk_out,
      vk_xk_out.mutable_data_ptr(),
      vk_xk_out.numel(),
      from_at_scalartype(vk_xk_out.scalar_type()));

  EXPECT_TRUE(at::allclose(xq_out, vk_xq_out, 1e-4, 1e-4));
  EXPECT_TRUE(at::allclose(xk_out, vk_xk_out, 1e-4, 1e-4));
}

TEST(VulkanRotaryEmbeddingTest, rotary_embedding_test) {
  test_reference();
}

TEST(VulkanRotaryEmbeddingTest, rotary_embedding_llama3_params_test) {
  test_reference(
      /*n_heads=*/32,
      /*n_kv_heads=*/8,
      /*dim=*/2048);
}

TEST(VulkanRotaryEmbeddingTest, rotary_embedding_llama3_params_test_seq_len_3) {
  test_reference(
      /*n_heads=*/32,
      /*n_kv_heads=*/8,
      /*dim=*/2048,
      /*seq_len=*/3);
}

//
// HuggingFace RoPE reference and tests
//

std::pair<at::Tensor, at::Tensor> rotary_embedding_hf_impl(
    const at::Tensor& xq,
    const at::Tensor& xk,
    const at::Tensor& freqs_cos,
    const at::Tensor& freqs_sin) {
  const int64_t head_dim = xq.size(3);
  const int64_t half_dim = head_dim / 2;

  // Split into first half and second half along head_dim
  at::Tensor xq_first = xq.slice(/*dim=*/3, /*start=*/0, /*end=*/half_dim);
  at::Tensor xq_second = xq.slice(/*dim=*/3, /*start=*/half_dim);
  at::Tensor xk_first = xk.slice(/*dim=*/3, /*start=*/0, /*end=*/half_dim);
  at::Tensor xk_second = xk.slice(/*dim=*/3, /*start=*/half_dim);

  // freqs are (seq_len, head_dim) but duplicated; use first half only
  at::Tensor cos_half =
      freqs_cos.slice(/*dim=*/1, /*start=*/0, /*end=*/half_dim);
  at::Tensor sin_half =
      freqs_sin.slice(/*dim=*/1, /*start=*/0, /*end=*/half_dim);

  at::Tensor cos_reshape =
      cos_half.reshape({1, cos_half.size(0), 1, cos_half.size(1)});
  at::Tensor sin_reshape =
      sin_half.reshape({1, sin_half.size(0), 1, sin_half.size(1)});

  // out[i]     = x[i] * cos[i] - x[i+D/2] * sin[i]
  // out[i+D/2] = x[i+D/2] * cos[i] + x[i] * sin[i]
  at::Tensor xq_out_first = xq_first * cos_reshape - xq_second * sin_reshape;
  at::Tensor xq_out_second = xq_second * cos_reshape + xq_first * sin_reshape;
  at::Tensor xk_out_first = xk_first * cos_reshape - xk_second * sin_reshape;
  at::Tensor xk_out_second = xk_second * cos_reshape + xk_first * sin_reshape;

  at::Tensor xq_out = at::cat({xq_out_first, xq_out_second}, /*dim=*/3);
  at::Tensor xk_out = at::cat({xk_out_first, xk_out_second}, /*dim=*/3);

  return std::make_pair(xq_out, xk_out);
}

void test_reference_hf(
    const int n_heads = 4,
    const int n_kv_heads = 2,
    const int dim = 32,
    const int seq_len = 1) {
  const int head_dim = dim / n_heads;

  at::Tensor xq = at::rand(
      {1, seq_len, n_heads, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor xk = at::rand(
      {1, seq_len, n_kv_heads, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));
  // HF convention: freqs are full head_dim (duplicated)
  at::Tensor freqs_cos =
      at::rand({seq_len, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor freqs_sin =
      at::rand({seq_len, head_dim}, at::device(at::kCPU).dtype(at::kFloat));

  std::pair<at::Tensor, at::Tensor> outs =
      rotary_embedding_hf_impl(xq, xk, freqs_cos, freqs_sin);
  at::Tensor& xq_out = outs.first;
  at::Tensor& xk_out = outs.second;

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

  IOValueRef r_xq = graph.add_input_tensor(
      xq.sizes().vec(), from_at_scalartype(xq.scalar_type()));
  IOValueRef r_xk = graph.add_input_tensor(
      xk.sizes().vec(), from_at_scalartype(xk.scalar_type()));
  IOValueRef r_freqs_cos = graph.add_input_tensor(
      freqs_cos.sizes().vec(), from_at_scalartype(freqs_cos.scalar_type()));
  IOValueRef r_freqs_sin = graph.add_input_tensor(
      freqs_sin.sizes().vec(), from_at_scalartype(freqs_sin.scalar_type()));

  const ValueRef r_xq_out = graph.add_tensor(
      xq_out.sizes().vec(), from_at_scalartype(xq_out.scalar_type()));
  const ValueRef r_xk_out = graph.add_tensor(
      xk_out.sizes().vec(), from_at_scalartype(xk_out.scalar_type()));

  const ValueRef r_start_pos = graph.add_scalar<int64_t>(0);

  VK_GET_OP_FN("et_vk.apply_rotary_emb_hf.default")
  (graph,
   {r_xq.value,
    r_xk.value,
    r_freqs_cos.value,
    r_freqs_sin.value,
    r_start_pos,
    graph.add_value_list({r_xq_out, r_xk_out})});

  ValueRef staging_xq_out = graph.set_output_tensor(r_xq_out);
  ValueRef staging_xk_out = graph.set_output_tensor(r_xk_out);

  graph.prepare();
  graph.prepack();

  graph.propagate_resize();
  graph.maybe_cast_and_copy_into_staging(
      r_xq.staging,
      xq.const_data_ptr(),
      xq.numel(),
      from_at_scalartype(xq.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_xk.staging,
      xk.const_data_ptr(),
      xk.numel(),
      from_at_scalartype(xk.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_freqs_cos.staging,
      freqs_cos.const_data_ptr(),
      freqs_cos.numel(),
      from_at_scalartype(freqs_cos.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_freqs_sin.staging,
      freqs_sin.const_data_ptr(),
      freqs_sin.numel(),
      from_at_scalartype(freqs_sin.scalar_type()));

  graph.execute();

  at::Tensor vk_xq_out = at::empty_like(xq_out);
  graph.maybe_cast_and_copy_from_staging(
      staging_xq_out,
      vk_xq_out.mutable_data_ptr(),
      vk_xq_out.numel(),
      from_at_scalartype(vk_xq_out.scalar_type()));

  at::Tensor vk_xk_out = at::empty_like(xk_out);
  graph.maybe_cast_and_copy_from_staging(
      staging_xk_out,
      vk_xk_out.mutable_data_ptr(),
      vk_xk_out.numel(),
      from_at_scalartype(vk_xk_out.scalar_type()));

  EXPECT_TRUE(at::allclose(xq_out, vk_xq_out, 1e-4, 1e-4));
  EXPECT_TRUE(at::allclose(xk_out, vk_xk_out, 1e-4, 1e-4));
}

TEST(VulkanRotaryEmbeddingHFTest, rotary_embedding_hf_test) {
  test_reference_hf();
}

TEST(VulkanRotaryEmbeddingHFTest, rotary_embedding_hf_llama3_params_test) {
  test_reference_hf(
      /*n_heads=*/32,
      /*n_kv_heads=*/8,
      /*dim=*/2048);
}

TEST(
    VulkanRotaryEmbeddingHFTest,
    rotary_embedding_hf_llama3_params_test_seq_len_3) {
  test_reference_hf(
      /*n_heads=*/32,
      /*n_kv_heads=*/8,
      /*dim=*/2048,
      /*seq_len=*/3);
}

TEST(VulkanRotaryEmbeddingHFTest, rotary_embedding_hf_head_dim_128) {
  test_reference_hf(
      /*n_heads=*/8,
      /*n_kv_heads=*/4,
      /*dim=*/1024,
      /*seq_len=*/5);
}

// Tests dynamic resize from prefill (seq_len=N) to decode (seq_len=1),
// simulating the actual LLM inference pattern that was previously broken.
TEST(VulkanRotaryEmbeddingHFTest, rotary_embedding_hf_dynamic_resize_qwen3) {
  const int n_heads = 16;
  const int n_kv_heads = 8;
  const int head_dim = 128;
  const int prefill_seq_len = 7;

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

  // Build graph with prefill shapes (max size)
  IOValueRef r_xq = graph.add_input_tensor(
      {1, prefill_seq_len, n_heads, head_dim}, vkapi::kFloat);
  IOValueRef r_xk = graph.add_input_tensor(
      {1, prefill_seq_len, n_kv_heads, head_dim}, vkapi::kFloat);
  IOValueRef r_freqs_cos =
      graph.add_input_tensor({prefill_seq_len, head_dim}, vkapi::kFloat);
  IOValueRef r_freqs_sin =
      graph.add_input_tensor({prefill_seq_len, head_dim}, vkapi::kFloat);

  const ValueRef r_xq_out =
      graph.add_tensor({1, prefill_seq_len, n_heads, head_dim}, vkapi::kFloat);
  const ValueRef r_xk_out = graph.add_tensor(
      {1, prefill_seq_len, n_kv_heads, head_dim}, vkapi::kFloat);

  const ValueRef r_start_pos = graph.add_scalar<int64_t>(0);

  VK_GET_OP_FN("et_vk.apply_rotary_emb_hf.default")
  (graph,
   {r_xq.value,
    r_xk.value,
    r_freqs_cos.value,
    r_freqs_sin.value,
    r_start_pos,
    graph.add_value_list({r_xq_out, r_xk_out})});

  ValueRef staging_xq_out = graph.set_output_tensor(r_xq_out);
  ValueRef staging_xk_out = graph.set_output_tensor(r_xk_out);

  graph.prepare();
  graph.prepack();

  // --- Prefill run (seq_len = 7) ---
  {
    at::Tensor xq = at::rand(
        {1, prefill_seq_len, n_heads, head_dim},
        at::device(at::kCPU).dtype(at::kFloat));
    at::Tensor xk = at::rand(
        {1, prefill_seq_len, n_kv_heads, head_dim},
        at::device(at::kCPU).dtype(at::kFloat));
    at::Tensor freqs_cos = at::rand(
        {prefill_seq_len, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
    at::Tensor freqs_sin = at::rand(
        {prefill_seq_len, head_dim}, at::device(at::kCPU).dtype(at::kFloat));

    auto ref = rotary_embedding_hf_impl(xq, xk, freqs_cos, freqs_sin);

    graph.resize_input(0, xq.sizes().vec());
    graph.resize_input(1, xk.sizes().vec());
    graph.resize_input(2, freqs_cos.sizes().vec());
    graph.resize_input(3, freqs_sin.sizes().vec());
    graph.propagate_resize();

    graph.maybe_cast_and_copy_into_staging(
        r_xq.staging, xq.const_data_ptr(), xq.numel(), vkapi::kFloat);
    graph.maybe_cast_and_copy_into_staging(
        r_xk.staging, xk.const_data_ptr(), xk.numel(), vkapi::kFloat);
    graph.maybe_cast_and_copy_into_staging(
        r_freqs_cos.staging,
        freqs_cos.const_data_ptr(),
        freqs_cos.numel(),
        vkapi::kFloat);
    graph.maybe_cast_and_copy_into_staging(
        r_freqs_sin.staging,
        freqs_sin.const_data_ptr(),
        freqs_sin.numel(),
        vkapi::kFloat);

    graph.execute();

    at::Tensor vk_xq_out = at::empty_like(ref.first);
    graph.maybe_cast_and_copy_from_staging(
        staging_xq_out,
        vk_xq_out.mutable_data_ptr(),
        vk_xq_out.numel(),
        vkapi::kFloat);
    at::Tensor vk_xk_out = at::empty_like(ref.second);
    graph.maybe_cast_and_copy_from_staging(
        staging_xk_out,
        vk_xk_out.mutable_data_ptr(),
        vk_xk_out.numel(),
        vkapi::kFloat);

    EXPECT_TRUE(at::allclose(ref.first, vk_xq_out, 1e-4, 1e-4))
        << "Prefill xq_out mismatch";
    EXPECT_TRUE(at::allclose(ref.second, vk_xk_out, 1e-4, 1e-4))
        << "Prefill xk_out mismatch";
  }

  // --- Decode run (seq_len = 1) ---
  {
    at::Tensor xq = at::rand(
        {1, 1, n_heads, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
    at::Tensor xk = at::rand(
        {1, 1, n_kv_heads, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
    at::Tensor freqs_cos =
        at::rand({1, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
    at::Tensor freqs_sin =
        at::rand({1, head_dim}, at::device(at::kCPU).dtype(at::kFloat));

    auto ref = rotary_embedding_hf_impl(xq, xk, freqs_cos, freqs_sin);

    graph.resize_input(0, xq.sizes().vec());
    graph.resize_input(1, xk.sizes().vec());
    graph.resize_input(2, freqs_cos.sizes().vec());
    graph.resize_input(3, freqs_sin.sizes().vec());
    graph.propagate_resize();

    graph.maybe_cast_and_copy_into_staging(
        r_xq.staging, xq.const_data_ptr(), xq.numel(), vkapi::kFloat);
    graph.maybe_cast_and_copy_into_staging(
        r_xk.staging, xk.const_data_ptr(), xk.numel(), vkapi::kFloat);
    graph.maybe_cast_and_copy_into_staging(
        r_freqs_cos.staging,
        freqs_cos.const_data_ptr(),
        freqs_cos.numel(),
        vkapi::kFloat);
    graph.maybe_cast_and_copy_into_staging(
        r_freqs_sin.staging,
        freqs_sin.const_data_ptr(),
        freqs_sin.numel(),
        vkapi::kFloat);

    graph.execute();

    at::Tensor vk_xq_out = at::empty_like(ref.first);
    graph.maybe_cast_and_copy_from_staging(
        staging_xq_out,
        vk_xq_out.mutable_data_ptr(),
        vk_xq_out.numel(),
        vkapi::kFloat);
    at::Tensor vk_xk_out = at::empty_like(ref.second);
    graph.maybe_cast_and_copy_from_staging(
        staging_xk_out,
        vk_xk_out.mutable_data_ptr(),
        vk_xk_out.numel(),
        vkapi::kFloat);

    EXPECT_TRUE(at::allclose(ref.first, vk_xq_out, 1e-4, 1e-4))
        << "Decode xq_out mismatch";
    EXPECT_TRUE(at::allclose(ref.second, vk_xk_out, 1e-4, 1e-4))
        << "Decode xk_out mismatch";
  }
}

// Tests that start_pos correctly offsets into the full freqs table.
// The Vulkan op receives the full [max_seq_len, head_dim] freqs table plus a
// start_pos offset, while the reference impl receives pre-sliced freqs.
void test_reference_hf_with_start_pos(
    const int n_heads = 8,
    const int n_kv_heads = 4,
    const int head_dim = 128,
    const int seq_len = 3,
    const int start_pos = 7,
    const int max_seq_len = 32) {
  at::Tensor xq = at::rand(
      {1, seq_len, n_heads, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor xk = at::rand(
      {1, seq_len, n_kv_heads, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));

  // Full freqs table of size [max_seq_len, head_dim]
  at::Tensor freqs_cos_full =
      at::rand({max_seq_len, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor freqs_sin_full =
      at::rand({max_seq_len, head_dim}, at::device(at::kCPU).dtype(at::kFloat));

  // Slice freqs for the reference implementation
  at::Tensor freqs_cos_sliced =
      freqs_cos_full.slice(/*dim=*/0, start_pos, start_pos + seq_len);
  at::Tensor freqs_sin_sliced =
      freqs_sin_full.slice(/*dim=*/0, start_pos, start_pos + seq_len);

  // Reference uses pre-sliced freqs
  std::pair<at::Tensor, at::Tensor> ref =
      rotary_embedding_hf_impl(xq, xk, freqs_cos_sliced, freqs_sin_sliced);

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

  IOValueRef r_xq = graph.add_input_tensor(
      xq.sizes().vec(), from_at_scalartype(xq.scalar_type()));
  IOValueRef r_xk = graph.add_input_tensor(
      xk.sizes().vec(), from_at_scalartype(xk.scalar_type()));
  // Vulkan op receives full freqs table
  IOValueRef r_freqs_cos = graph.add_input_tensor(
      freqs_cos_full.sizes().vec(),
      from_at_scalartype(freqs_cos_full.scalar_type()));
  IOValueRef r_freqs_sin = graph.add_input_tensor(
      freqs_sin_full.sizes().vec(),
      from_at_scalartype(freqs_sin_full.scalar_type()));

  const ValueRef r_xq_out = graph.add_tensor(
      ref.first.sizes().vec(), from_at_scalartype(ref.first.scalar_type()));
  const ValueRef r_xk_out = graph.add_tensor(
      ref.second.sizes().vec(), from_at_scalartype(ref.second.scalar_type()));

  const ValueRef r_start_pos = graph.add_scalar<int64_t>(start_pos);

  VK_GET_OP_FN("et_vk.apply_rotary_emb_hf.default")
  (graph,
   {r_xq.value,
    r_xk.value,
    r_freqs_cos.value,
    r_freqs_sin.value,
    r_start_pos,
    graph.add_value_list({r_xq_out, r_xk_out})});

  ValueRef staging_xq_out = graph.set_output_tensor(r_xq_out);
  ValueRef staging_xk_out = graph.set_output_tensor(r_xk_out);

  graph.prepare();
  graph.prepack();

  graph.propagate_resize();
  graph.maybe_cast_and_copy_into_staging(
      r_xq.staging,
      xq.const_data_ptr(),
      xq.numel(),
      from_at_scalartype(xq.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_xk.staging,
      xk.const_data_ptr(),
      xk.numel(),
      from_at_scalartype(xk.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_freqs_cos.staging,
      freqs_cos_full.const_data_ptr(),
      freqs_cos_full.numel(),
      from_at_scalartype(freqs_cos_full.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_freqs_sin.staging,
      freqs_sin_full.const_data_ptr(),
      freqs_sin_full.numel(),
      from_at_scalartype(freqs_sin_full.scalar_type()));

  graph.execute();

  at::Tensor vk_xq_out = at::empty_like(ref.first);
  graph.maybe_cast_and_copy_from_staging(
      staging_xq_out,
      vk_xq_out.mutable_data_ptr(),
      vk_xq_out.numel(),
      from_at_scalartype(vk_xq_out.scalar_type()));

  at::Tensor vk_xk_out = at::empty_like(ref.second);
  graph.maybe_cast_and_copy_from_staging(
      staging_xk_out,
      vk_xk_out.mutable_data_ptr(),
      vk_xk_out.numel(),
      from_at_scalartype(vk_xk_out.scalar_type()));

  EXPECT_TRUE(at::allclose(ref.first, vk_xq_out, 1e-4, 1e-4));
  EXPECT_TRUE(at::allclose(ref.second, vk_xk_out, 1e-4, 1e-4));
}

TEST(VulkanRotaryEmbeddingHFTest, rotary_embedding_hf_start_pos_offset) {
  test_reference_hf_with_start_pos();
}

TEST(VulkanRotaryEmbeddingHFTest, rotary_embedding_hf_start_pos_decode) {
  test_reference_hf_with_start_pos(
      /*n_heads=*/16,
      /*n_kv_heads=*/8,
      /*head_dim=*/128,
      /*seq_len=*/1,
      /*start_pos=*/15,
      /*max_seq_len=*/64);
}

//
// Partial rotary tests (partial_rotary_factor < 1.0)
//

// Reference impl for partial rotary: only first rotary_dim elements are
// rotated, the rest pass through unchanged.
std::pair<at::Tensor, at::Tensor> rotary_embedding_hf_partial_impl(
    const at::Tensor& xq,
    const at::Tensor& xk,
    const at::Tensor& freqs_cos,
    const at::Tensor& freqs_sin) {
  const int64_t rotary_dim = freqs_cos.size(1);
  const int64_t rotary_half = rotary_dim / 2;

  // Split into rotary and passthrough regions
  at::Tensor xq_rot = xq.slice(/*dim=*/3, /*start=*/0, /*end=*/rotary_dim);
  at::Tensor xq_pass = xq.slice(/*dim=*/3, /*start=*/rotary_dim);
  at::Tensor xk_rot = xk.slice(/*dim=*/3, /*start=*/0, /*end=*/rotary_dim);
  at::Tensor xk_pass = xk.slice(/*dim=*/3, /*start=*/rotary_dim);

  // Split rotary region into first and second halves
  at::Tensor xq_first =
      xq_rot.slice(/*dim=*/3, /*start=*/0, /*end=*/rotary_half);
  at::Tensor xq_second = xq_rot.slice(/*dim=*/3, /*start=*/rotary_half);
  at::Tensor xk_first =
      xk_rot.slice(/*dim=*/3, /*start=*/0, /*end=*/rotary_half);
  at::Tensor xk_second = xk_rot.slice(/*dim=*/3, /*start=*/rotary_half);

  // freqs are (seq_len, rotary_dim); use first half only
  at::Tensor cos_half =
      freqs_cos.slice(/*dim=*/1, /*start=*/0, /*end=*/rotary_half);
  at::Tensor sin_half =
      freqs_sin.slice(/*dim=*/1, /*start=*/0, /*end=*/rotary_half);

  at::Tensor cos_reshape =
      cos_half.reshape({1, cos_half.size(0), 1, cos_half.size(1)});
  at::Tensor sin_reshape =
      sin_half.reshape({1, sin_half.size(0), 1, sin_half.size(1)});

  at::Tensor xq_out_first = xq_first * cos_reshape - xq_second * sin_reshape;
  at::Tensor xq_out_second = xq_second * cos_reshape + xq_first * sin_reshape;
  at::Tensor xk_out_first = xk_first * cos_reshape - xk_second * sin_reshape;
  at::Tensor xk_out_second = xk_second * cos_reshape + xk_first * sin_reshape;

  at::Tensor xq_out =
      at::cat({xq_out_first, xq_out_second, xq_pass}, /*dim=*/3);
  at::Tensor xk_out =
      at::cat({xk_out_first, xk_out_second, xk_pass}, /*dim=*/3);

  return std::make_pair(xq_out, xk_out);
}

void test_reference_hf_partial_rotary(
    const int n_heads = 8,
    const int n_kv_heads = 4,
    const int head_dim = 128,
    const int rotary_dim = 96,
    const int seq_len = 3,
    const int start_pos = 0,
    const int max_seq_len = 32) {
  at::Tensor xq = at::rand(
      {1, seq_len, n_heads, head_dim}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor xk = at::rand(
      {1, seq_len, n_kv_heads, head_dim},
      at::device(at::kCPU).dtype(at::kFloat));

  // Full freqs table with rotary_dim < head_dim
  at::Tensor freqs_cos_full = at::rand(
      {max_seq_len, rotary_dim}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor freqs_sin_full = at::rand(
      {max_seq_len, rotary_dim}, at::device(at::kCPU).dtype(at::kFloat));

  // Slice freqs for reference
  at::Tensor freqs_cos_sliced =
      freqs_cos_full.slice(/*dim=*/0, start_pos, start_pos + seq_len);
  at::Tensor freqs_sin_sliced =
      freqs_sin_full.slice(/*dim=*/0, start_pos, start_pos + seq_len);

  auto ref = rotary_embedding_hf_partial_impl(
      xq, xk, freqs_cos_sliced, freqs_sin_sliced);

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

  IOValueRef r_xq = graph.add_input_tensor(
      xq.sizes().vec(), from_at_scalartype(xq.scalar_type()));
  IOValueRef r_xk = graph.add_input_tensor(
      xk.sizes().vec(), from_at_scalartype(xk.scalar_type()));
  IOValueRef r_freqs_cos = graph.add_input_tensor(
      freqs_cos_full.sizes().vec(),
      from_at_scalartype(freqs_cos_full.scalar_type()));
  IOValueRef r_freqs_sin = graph.add_input_tensor(
      freqs_sin_full.sizes().vec(),
      from_at_scalartype(freqs_sin_full.scalar_type()));

  const ValueRef r_xq_out = graph.add_tensor(
      ref.first.sizes().vec(), from_at_scalartype(ref.first.scalar_type()));
  const ValueRef r_xk_out = graph.add_tensor(
      ref.second.sizes().vec(), from_at_scalartype(ref.second.scalar_type()));

  const ValueRef r_start_pos = graph.add_scalar<int64_t>(start_pos);

  VK_GET_OP_FN("et_vk.apply_rotary_emb_hf.default")
  (graph,
   {r_xq.value,
    r_xk.value,
    r_freqs_cos.value,
    r_freqs_sin.value,
    r_start_pos,
    graph.add_value_list({r_xq_out, r_xk_out})});

  ValueRef staging_xq_out = graph.set_output_tensor(r_xq_out);
  ValueRef staging_xk_out = graph.set_output_tensor(r_xk_out);

  graph.prepare();
  graph.prepack();

  graph.propagate_resize();
  graph.maybe_cast_and_copy_into_staging(
      r_xq.staging,
      xq.const_data_ptr(),
      xq.numel(),
      from_at_scalartype(xq.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_xk.staging,
      xk.const_data_ptr(),
      xk.numel(),
      from_at_scalartype(xk.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_freqs_cos.staging,
      freqs_cos_full.const_data_ptr(),
      freqs_cos_full.numel(),
      from_at_scalartype(freqs_cos_full.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_freqs_sin.staging,
      freqs_sin_full.const_data_ptr(),
      freqs_sin_full.numel(),
      from_at_scalartype(freqs_sin_full.scalar_type()));

  graph.execute();

  at::Tensor vk_xq_out = at::empty_like(ref.first);
  graph.maybe_cast_and_copy_from_staging(
      staging_xq_out,
      vk_xq_out.mutable_data_ptr(),
      vk_xq_out.numel(),
      from_at_scalartype(vk_xq_out.scalar_type()));

  at::Tensor vk_xk_out = at::empty_like(ref.second);
  graph.maybe_cast_and_copy_from_staging(
      staging_xk_out,
      vk_xk_out.mutable_data_ptr(),
      vk_xk_out.numel(),
      from_at_scalartype(vk_xk_out.scalar_type()));

  EXPECT_TRUE(at::allclose(ref.first, vk_xq_out, 1e-4, 1e-4));
  EXPECT_TRUE(at::allclose(ref.second, vk_xk_out, 1e-4, 1e-4));
}

// Phi4 Mini-like: head_dim=128, rotary_dim=96 (partial_rotary_factor=0.75)
TEST(VulkanRotaryEmbeddingHFTest, rotary_embedding_hf_partial_rotary) {
  test_reference_hf_partial_rotary();
}

// Partial rotary with non-zero start_pos
TEST(
    VulkanRotaryEmbeddingHFTest,
    rotary_embedding_hf_partial_rotary_start_pos) {
  test_reference_hf_partial_rotary(
      /*n_heads=*/16,
      /*n_kv_heads=*/8,
      /*head_dim=*/128,
      /*rotary_dim=*/96,
      /*seq_len=*/1,
      /*start_pos=*/10,
      /*max_seq_len=*/64);
}
