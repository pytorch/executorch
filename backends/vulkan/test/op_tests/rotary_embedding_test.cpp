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
