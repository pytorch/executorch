/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <iostream>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include "test_utils.h"

namespace {

void test_vulkan_update_cache_with_indices(
    vkcompute::utils::StorageType cache_storage_type,
    vkcompute::utils::StorageType value_storage_type,
    at::ScalarType dtype) {
  using namespace vkcompute;

  at::Tensor cache =
      at::full({1, 8, 3, 6}, -1, at::device(at::kCPU).dtype(dtype));
  at::Tensor value_1 =
      at::arange(1, 1 + 1 * 4 * 3 * 6, at::device(at::kCPU).dtype(at::kFloat))
          .reshape({1, 4, 3, 6})
          .to(dtype);
  at::Tensor indices_1 =
      at::tensor({6, 7, 0, 1}, at::device(at::kCPU).dtype(at::kLong))
          .reshape({1, 4});
  at::Tensor value_2 =
      at::arange(
          101, 101 + 1 * 2 * 3 * 6, at::device(at::kCPU).dtype(at::kFloat))
          .reshape({1, 2, 3, 6})
          .to(dtype);
  at::Tensor indices_2 =
      at::tensor({1, 4}, at::device(at::kCPU).dtype(at::kLong)).reshape({1, 2});
  at::Tensor value_3 =
      at::arange(
          201, 201 + 1 * 3 * 3 * 6, at::device(at::kCPU).dtype(at::kFloat))
          .reshape({1, 3, 3, 6})
          .to(dtype);
  at::Tensor indices_3 =
      at::tensor({2, -1, 99}, at::device(at::kCPU).dtype(at::kLong))
          .reshape({1, 3});

  at::Tensor expected = cache.clone();
  for (int64_t s = 0; s < indices_1.size(1); ++s) {
    expected[0][indices_1[0][s].item<int64_t>()].copy_(value_1[0][s]);
  }
  for (int64_t s = 0; s < indices_2.size(1); ++s) {
    expected[0][indices_2[0][s].item<int64_t>()].copy_(value_2[0][s]);
  }
  expected[0][2].copy_(value_3[0][0]);

  GraphConfig config;
  ComputeGraph graph(config);
  IOValueRef r_cache = graph.add_input_tensor(
      cache.sizes().vec(), from_at_scalartype(dtype), cache_storage_type);
  IOValueRef r_value_1 = graph.add_input_tensor(
      value_1.sizes().vec(), from_at_scalartype(dtype), value_storage_type);
  IOValueRef r_indices_1 = graph.add_input_tensor(
      indices_1.sizes().vec(), vkapi::kInt, utils::kBuffer);
  IOValueRef r_value_2 = graph.add_input_tensor(
      value_2.sizes().vec(), from_at_scalartype(dtype), value_storage_type);
  IOValueRef r_indices_2 = graph.add_input_tensor(
      indices_2.sizes().vec(), vkapi::kInt, utils::kBuffer);
  IOValueRef r_value_3 = graph.add_input_tensor(
      value_3.sizes().vec(), from_at_scalartype(dtype), value_storage_type);
  IOValueRef r_indices_3 = graph.add_input_tensor(
      indices_3.sizes().vec(), vkapi::kInt, utils::kBuffer);
  const ValueRef r_start_pos = graph.add_symint(37);
  const ValueRef r_dummy_out_1 =
      graph.add_tensor({1}, from_at_scalartype(dtype), utils::kBuffer);
  const ValueRef r_dummy_out_2 =
      graph.add_tensor({1}, from_at_scalartype(dtype), utils::kBuffer);
  const ValueRef r_dummy_out_3 =
      graph.add_tensor({1}, from_at_scalartype(dtype), utils::kBuffer);

  VK_GET_OP_FN("update_cache_with_indices.default")
  (graph,
   {r_value_1.value,
    r_cache.value,
    r_start_pos,
    r_indices_1.value,
    r_dummy_out_1});
  VK_GET_OP_FN("update_cache_with_indices.default")
  (graph,
   {r_value_2.value,
    r_cache.value,
    r_start_pos,
    r_indices_2.value,
    r_dummy_out_2});
  VK_GET_OP_FN("update_cache_with_indices.default")
  (graph,
   {r_value_3.value,
    r_cache.value,
    r_start_pos,
    r_indices_3.value,
    r_dummy_out_3});

  const ValueRef staging_out = graph.set_output_tensor(r_cache.value);
  graph.prepare();
  graph.prepack();

  graph.maybe_cast_and_copy_into_staging(
      r_cache.staging,
      cache.const_data_ptr(),
      cache.numel(),
      from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_value_1.staging,
      value_1.const_data_ptr(),
      value_1.numel(),
      from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_indices_1.staging,
      indices_1.const_data_ptr(),
      indices_1.numel(),
      vkapi::kLong);
  graph.maybe_cast_and_copy_into_staging(
      r_value_2.staging,
      value_2.const_data_ptr(),
      value_2.numel(),
      from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_indices_2.staging,
      indices_2.const_data_ptr(),
      indices_2.numel(),
      vkapi::kLong);
  graph.maybe_cast_and_copy_into_staging(
      r_value_3.staging,
      value_3.const_data_ptr(),
      value_3.numel(),
      from_at_scalartype(dtype));
  graph.maybe_cast_and_copy_into_staging(
      r_indices_3.staging,
      indices_3.const_data_ptr(),
      indices_3.numel(),
      vkapi::kLong);

  graph.execute();

  at::Tensor actual = at::empty_like(cache).contiguous();
  graph.maybe_cast_and_copy_from_staging(
      staging_out,
      actual.mutable_data_ptr(),
      actual.numel(),
      from_at_scalartype(dtype));
  if (!at::equal(expected, actual)) {
    const at::Tensor mismatch = expected.ne(actual).flatten();
    const int64_t first = mismatch.nonzero()[0].item<int64_t>();
    std::cout << "first mismatch " << first << ": expected "
              << expected.flatten()[first].item() << ", actual "
              << actual.flatten()[first].item() << std::endl;
  }
  ASSERT_TRUE(at::equal(expected, actual));
}

} // namespace

TEST(VulkanUpdateCacheWithIndicesTest, buffer_fp32) {
  test_vulkan_update_cache_with_indices(
      vkcompute::utils::kBuffer, vkcompute::utils::kBuffer, at::kFloat);
}

TEST(VulkanUpdateCacheWithIndicesTest, texture_fp32) {
  test_vulkan_update_cache_with_indices(
      vkcompute::utils::kTexture3D, vkcompute::utils::kTexture3D, at::kFloat);
}

TEST(VulkanUpdateCacheWithIndicesTest, buffer_fp16) {
  test_vulkan_update_cache_with_indices(
      vkcompute::utils::kBuffer, vkcompute::utils::kBuffer, at::kHalf);
}

TEST(VulkanUpdateCacheWithIndicesTest, texture_fp16) {
  test_vulkan_update_cache_with_indices(
      vkcompute::utils::kTexture3D, vkcompute::utils::kTexture3D, at::kHalf);
}

TEST(VulkanUpdateCacheWithIndicesTest, texture_cache_buffer_value_fp32) {
  test_vulkan_update_cache_with_indices(
      vkcompute::utils::kTexture3D, vkcompute::utils::kBuffer, at::kFloat);
}

TEST(VulkanUpdateCacheWithIndicesTest, buffer_cache_texture_value_fp32) {
  test_vulkan_update_cache_with_indices(
      vkcompute::utils::kBuffer, vkcompute::utils::kTexture3D, at::kFloat);
}

TEST(VulkanUpdateCacheWithIndicesTest, texture_cache_buffer_value_fp16) {
  test_vulkan_update_cache_with_indices(
      vkcompute::utils::kTexture3D, vkcompute::utils::kBuffer, at::kHalf);
}

TEST(VulkanUpdateCacheWithIndicesTest, buffer_cache_texture_value_fp16) {
  test_vulkan_update_cache_with_indices(
      vkcompute::utils::kBuffer, vkcompute::utils::kTexture3D, at::kHalf);
}
