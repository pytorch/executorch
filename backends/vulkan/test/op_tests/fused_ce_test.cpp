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

#include <iostream>
#include <vector>

//
// Reference implementation: transcription of the CPU-eager fused_ce_impl
// (backends/vulkan/custom_ops_lib.py), computed with ATen (library golden).
//

std::pair<at::Tensor, at::Tensor> fused_ce_reference(
    const at::Tensor& logits,
    const at::Tensor& labels,
    double n_valid) {
  at::Tensor mask = labels.ge(0);
  at::Tensor safe = labels.clamp_min(0).to(at::kLong);
  at::Tensor lse = at::logsumexp(logits, -1);
  at::Tensor picked = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1);
  at::Tensor loss =
      at::where(mask, (lse - picked) / n_valid, at::zeros_like(lse)).sum();
  at::Tensor softmax = at::softmax(logits, -1);
  at::Tensor onehot = at::one_hot(safe, logits.size(-1)).to(logits.dtype());
  at::Tensor dlogits = at::where(
      mask.unsqueeze(-1),
      (softmax - onehot) / n_valid,
      at::zeros_like(softmax));
  return {loss, dlogits};
}

void test_vulkan_fused_ce(
    const int n_rows,
    const int vocab,
    const std::vector<int32_t>& labels_data,
    const double n_valid) {
  torch::manual_seed(0);

  at::Tensor logits =
      at::rand({n_rows, vocab}, at::device(at::kCPU).dtype(at::kFloat)) * 4.0 -
      2.0;
  at::Tensor labels =
      at::from_blob(
          const_cast<int32_t*>(labels_data.data()), {n_rows}, at::kInt)
          .clone();

  auto ref = fused_ce_reference(logits, labels, n_valid);
  at::Tensor ref_loss = ref.first;
  at::Tensor ref_dlogits = ref.second;

  using namespace vkcompute;

  GraphConfig config;
  ComputeGraph graph(config);

  IOValueRef r_logits = graph.add_input_tensor(
      logits.sizes().vec(), vkapi::kFloat, utils::kBuffer);
  IOValueRef r_labels =
      graph.add_input_tensor(labels.sizes().vec(), vkapi::kInt, utils::kBuffer);

  const ValueRef r_n_valid = graph.add_scalar<double>(n_valid);

  const ValueRef r_loss = graph.add_tensor({}, vkapi::kFloat, utils::kBuffer);
  const ValueRef r_dlogits =
      graph.add_tensor({n_rows, vocab}, vkapi::kFloat, utils::kBuffer);
  const ValueRef r_out = graph.add_value_list({r_loss, r_dlogits});

  VK_GET_OP_FN("et_vk.fused_ce.default")
  (graph, {r_logits.value, r_labels.value, r_n_valid, r_out});

  ValueRef staging_loss = graph.set_output_tensor(r_loss);
  ValueRef staging_dlogits = graph.set_output_tensor(r_dlogits);

  graph.prepare();
  graph.prepack();
  graph.propagate_resize();

  graph.maybe_cast_and_copy_into_staging(
      r_logits.staging, logits.const_data_ptr(), logits.numel(), vkapi::kFloat);
  graph.maybe_cast_and_copy_into_staging(
      r_labels.staging, labels.const_data_ptr(), labels.numel(), vkapi::kInt);

  graph.execute();

  at::Tensor vk_loss = at::zeros({}, at::device(at::kCPU).dtype(at::kFloat));
  graph.maybe_cast_and_copy_from_staging(
      staging_loss, vk_loss.mutable_data_ptr(), 1, vkapi::kFloat);

  at::Tensor vk_dlogits =
      at::zeros({n_rows, vocab}, at::device(at::kCPU).dtype(at::kFloat));
  graph.maybe_cast_and_copy_from_staging(
      staging_dlogits,
      vk_dlogits.mutable_data_ptr(),
      vk_dlogits.numel(),
      vkapi::kFloat);

  const double atol = 1e-4;
  const double rtol = 1e-4;

  const bool loss_ok = at::allclose(ref_loss, vk_loss, rtol, atol);
  const bool dlogits_ok = at::allclose(ref_dlogits, vk_dlogits, rtol, atol);

  if (!loss_ok || !dlogits_ok) {
    std::cout << "fused_ce mismatch: n_rows=" << n_rows << " vocab=" << vocab
              << " n_valid=" << n_valid << std::endl;
    std::cout << "loss ref=" << ref_loss.item<float>()
              << " vk=" << vk_loss.item<float>() << std::endl;
    std::cout << "max dlogits diff="
              << at::max(at::abs(ref_dlogits - vk_dlogits)).item<float>()
              << std::endl;
  }
  ASSERT_TRUE(loss_ok);
  ASSERT_TRUE(dlogits_ok);
}

TEST(VulkanFusedCeTest, all_valid_small) {
  test_vulkan_fused_ce(
      /*n_rows=*/4, /*vocab=*/8, /*labels=*/{0, 3, 7, 1}, /*n_valid=*/4.0);
}

TEST(VulkanFusedCeTest, masked_label) {
  // One masked row (label < 0): contributes 0 to loss and 0 gradient.
  test_vulkan_fused_ce(
      /*n_rows=*/4, /*vocab=*/8, /*labels=*/{2, -1, 5, 0}, /*n_valid=*/3.0);
}

TEST(VulkanFusedCeTest, large_vocab_strided_reduce) {
  // vocab >> NWORKERS exercises the strided per-row reduction.
  test_vulkan_fused_ce(
      /*n_rows=*/3, /*vocab=*/200, /*labels=*/{17, -1, 199}, /*n_valid=*/2.0);
}
