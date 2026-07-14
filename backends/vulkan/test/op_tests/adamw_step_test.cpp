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

#include <cmath>
#include <tuple>

//
// Reference implementation: an ATen transcription of the CPU-eager
// adamw_step_impl golden in backends/vulkan/custom_ops_lib.py.
//

std::tuple<at::Tensor, at::Tensor, at::Tensor> adamw_reference_impl(
    const at::Tensor& param_in,
    const at::Tensor& m_in,
    const at::Tensor& v_in,
    const at::Tensor& grad,
    const double lr,
    const double beta1,
    const double beta2,
    const double eps,
    const double weight_decay,
    const double bias_correction1,
    const double bias_correction2) {
  at::Tensor param = param_in.clone();
  at::Tensor m = m_in.clone();
  at::Tensor v = v_in.clone();

  param.mul_(1.0 - lr * weight_decay);
  m.mul_(beta1).add_(grad, 1.0 - beta1);
  v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);
  at::Tensor mhat = m / bias_correction1;
  at::Tensor denom = (v / bias_correction2).sqrt() + eps;
  param.addcdiv_(mhat, denom, -lr);

  return std::make_tuple(param, m, v);
}

//
// Test harness
//

void test_vulkan_adamw_step(
    const std::vector<int64_t>& sizes,
    const double lr,
    const double beta1,
    const double beta2,
    const double eps,
    const double weight_decay,
    const int step) {
  torch::manual_seed(0);

  const double bias_correction1 = 1.0 - std::pow(beta1, step);
  const double bias_correction2 = 1.0 - std::pow(beta2, step);

  const auto options = at::device(at::kCPU).dtype(at::kFloat);
  at::Tensor param = at::randn(sizes, options);
  at::Tensor m = at::randn(sizes, options);
  at::Tensor v = at::rand(sizes, options); // second moment is non-negative
  at::Tensor grad = at::randn(sizes, options);

  at::Tensor ref_param;
  at::Tensor ref_m;
  at::Tensor ref_v;
  std::tie(ref_param, ref_m, ref_v) = adamw_reference_impl(
      param,
      m,
      v,
      grad,
      lr,
      beta1,
      beta2,
      eps,
      weight_decay,
      bias_correction1,
      bias_correction2);

  using namespace vkcompute;

  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef r_param = graph.add_tensor(sizes, vkapi::kFloat, utils::kBuffer);
  ValueRef r_m = graph.add_tensor(sizes, vkapi::kFloat, utils::kBuffer);
  ValueRef r_v = graph.add_tensor(sizes, vkapi::kFloat, utils::kBuffer);
  ValueRef r_grad = graph.add_tensor(sizes, vkapi::kFloat, utils::kBuffer);

  ValueRef staging_param = graph.set_input_tensor(r_param);
  ValueRef staging_m = graph.set_input_tensor(r_m);
  ValueRef staging_v = graph.set_input_tensor(r_v);
  ValueRef staging_grad = graph.set_input_tensor(r_grad);

  VK_GET_OP_FN("et_vk.adamw_step.default")
  (graph,
   {
       r_param,
       r_m,
       r_v,
       r_grad,
       graph.add_scalar<double>(lr),
       graph.add_scalar<double>(beta1),
       graph.add_scalar<double>(beta2),
       graph.add_scalar<double>(eps),
       graph.add_scalar<double>(weight_decay),
       graph.add_scalar<double>(bias_correction1),
       graph.add_scalar<double>(bias_correction2),
   });

  ValueRef staging_param_out = graph.set_output_tensor(r_param);
  ValueRef staging_m_out = graph.set_output_tensor(r_m);
  ValueRef staging_v_out = graph.set_output_tensor(r_v);

  graph.prepare();
  graph.prepack();

  graph.maybe_cast_and_copy_into_staging(
      staging_param, param.const_data_ptr(), param.numel(), vkapi::kFloat);
  graph.maybe_cast_and_copy_into_staging(
      staging_m, m.const_data_ptr(), m.numel(), vkapi::kFloat);
  graph.maybe_cast_and_copy_into_staging(
      staging_v, v.const_data_ptr(), v.numel(), vkapi::kFloat);
  graph.maybe_cast_and_copy_into_staging(
      staging_grad, grad.const_data_ptr(), grad.numel(), vkapi::kFloat);

  graph.execute();

  at::Tensor vk_param = at::empty_like(param);
  at::Tensor vk_m = at::empty_like(m);
  at::Tensor vk_v = at::empty_like(v);

  graph.maybe_cast_and_copy_from_staging(
      staging_param_out,
      vk_param.mutable_data_ptr(),
      vk_param.numel(),
      vkapi::kFloat);
  graph.maybe_cast_and_copy_from_staging(
      staging_m_out, vk_m.mutable_data_ptr(), vk_m.numel(), vkapi::kFloat);
  graph.maybe_cast_and_copy_from_staging(
      staging_v_out, vk_v.mutable_data_ptr(), vk_v.numel(), vkapi::kFloat);

  ASSERT_TRUE(at::allclose(ref_param, vk_param, /*rtol=*/1e-4, /*atol=*/1e-4));
  ASSERT_TRUE(at::allclose(ref_m, vk_m, /*rtol=*/1e-4, /*atol=*/1e-4));
  ASSERT_TRUE(at::allclose(ref_v, vk_v, /*rtol=*/1e-4, /*atol=*/1e-4));
}

TEST(VulkanAdamwStepTest, test_adamw_step_no_weight_decay) {
  test_vulkan_adamw_step(
      /*sizes=*/{4, 8},
      /*lr=*/1e-3,
      /*beta1=*/0.9,
      /*beta2=*/0.999,
      /*eps=*/1e-8,
      /*weight_decay=*/0.0,
      /*step=*/1);
}

TEST(VulkanAdamwStepTest, test_adamw_step_with_weight_decay) {
  test_vulkan_adamw_step(
      /*sizes=*/{3, 5, 7},
      /*lr=*/1e-2,
      /*beta1=*/0.9,
      /*beta2=*/0.999,
      /*eps=*/1e-8,
      /*weight_decay=*/0.01,
      /*step=*/10);
}

TEST(VulkanAdamwStepTest, test_adamw_step_large_1d) {
  test_vulkan_adamw_step(
      /*sizes=*/{1000},
      /*lr=*/1e-3,
      /*beta1=*/0.9,
      /*beta2=*/0.999,
      /*eps=*/1e-8,
      /*weight_decay=*/0.05,
      /*step=*/5);
}
