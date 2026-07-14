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

//
// Reference Implementation
//

// Golden: dW[N, K] = d_out^T @ x, contracting over the flattened leading dims.
// Mirrors the CPU-eager linear_q4gsw_dw_impl in custom_ops_lib.py.
at::Tensor linear_q4gsw_dw_reference_impl(
    const at::Tensor& d_out,
    const at::Tensor& x) {
  const int64_t N = d_out.size(-1);
  const int64_t K = x.size(-1);
  return d_out.reshape({-1, N})
      .t()
      .matmul(x.reshape({-1, K}))
      .contiguous();
}

//
// Test function
//

void test_vulkan_linear_q4gsw_dw_impl(
    const std::vector<int64_t>& d_out_sizes,
    const std::vector<int64_t>& x_sizes,
    const vkcompute::utils::StorageType storage =
        vkcompute::utils::kBuffer) {
  at::Tensor d_out =
      at::rand(d_out_sizes, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor x = at::rand(x_sizes, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor dW_ref = linear_q4gsw_dw_reference_impl(d_out, x);

  // Build Vulkan graph
  using namespace vkcompute;

  GraphConfig config;
  ComputeGraph graph(config);

  IOValueRef r_d_out = graph.add_input_tensor(
      d_out.sizes().vec(), from_at_scalartype(d_out.scalar_type()), storage);
  IOValueRef r_x = graph.add_input_tensor(
      x.sizes().vec(), from_at_scalartype(x.scalar_type()), storage);

  const ValueRef r_dW = graph.add_tensor(
      dW_ref.sizes().vec(), from_at_scalartype(dW_ref.scalar_type()), storage);

  VK_GET_OP_FN("et_vk.linear_q4gsw_dw.default")
  (graph, {r_d_out.value, r_x.value, r_dW});

  ValueRef staging_out = graph.set_output_tensor(r_dW);

  graph.prepare();
  graph.prepack();
  graph.propagate_resize();

  graph.maybe_cast_and_copy_into_staging(
      r_d_out.staging,
      d_out.const_data_ptr(),
      d_out.numel(),
      from_at_scalartype(d_out.scalar_type()));
  graph.maybe_cast_and_copy_into_staging(
      r_x.staging,
      x.const_data_ptr(),
      x.numel(),
      from_at_scalartype(x.scalar_type()));

  graph.execute();

  at::Tensor vk_dW = at::empty_like(dW_ref);
  graph.maybe_cast_and_copy_from_staging(
      staging_out,
      vk_dW.mutable_data_ptr(),
      vk_dW.numel(),
      from_at_scalartype(vk_dW.scalar_type()));

  ASSERT_TRUE(at::allclose(vk_dW, dW_ref, 1e-3, 1e-3));
}

// Tile-aligned 2D shapes (M, N, K all multiples of 4).
TEST(VulkanLinearQ4gswDwTest, test_tile_aligned) {
  test_vulkan_linear_q4gsw_dw_impl(
      /*d_out_sizes=*/{8, 16}, /*x_sizes=*/{8, 32});
}

// Non-tile-multiple shapes (M, N, K each not a multiple of 4) to exercise the
// partial-tile min()-clamp paths in the shader.
TEST(VulkanLinearQ4gswDwTest, test_non_tile_multiple) {
  test_vulkan_linear_q4gsw_dw_impl(
      /*d_out_sizes=*/{5, 6}, /*x_sizes=*/{5, 10});
}

// Leading dims > 2D: M is the flattened product of all leading dims.
TEST(VulkanLinearQ4gswDwTest, test_leading_dims_flatten) {
  test_vulkan_linear_q4gsw_dw_impl(
      /*d_out_sizes=*/{2, 3, 16}, /*x_sizes=*/{2, 3, 32});
}
