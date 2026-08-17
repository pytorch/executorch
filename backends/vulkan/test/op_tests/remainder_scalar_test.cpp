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

void test_vulkan_remainder_scalar(
    vkcompute::utils::StorageType storage_type,
    int64_t divisor) {
  using namespace vkcompute;

  at::Tensor input_1 =
      at::tensor(
          {-129, -128, -127, -1, 0, 1, 60, 67, 127, 128, 129, 255},
          at::device(at::kCPU).dtype(at::kInt))
          .reshape({2, 6});
  at::Tensor input_2 =
      at::tensor(
          {256, 257, -257, 383, 384, 385, 61, 68, 126, 130, -2, 2},
          at::device(at::kCPU).dtype(at::kInt))
          .reshape({2, 6});

  GraphConfig config;
  ComputeGraph graph(config);
  IOValueRef input =
      graph.add_input_tensor(input_1.sizes().vec(), vkapi::kInt, storage_type);
  const ValueRef scalar = graph.add_scalar<int64_t>(divisor);
  const ValueRef output =
      graph.add_tensor(input_1.sizes().vec(), vkapi::kInt, storage_type);

  VK_GET_OP_FN("aten.remainder.Scalar")
  (graph, {input.value, scalar, output});

  const ValueRef staging_output = graph.set_output_tensor(output);
  graph.prepare();
  graph.prepack();

  for (const at::Tensor& input_tensor : {input_1, input_2}) {
    graph.maybe_cast_and_copy_into_staging(
        input.staging,
        input_tensor.const_data_ptr(),
        input_tensor.numel(),
        vkapi::kInt);
    graph.execute();

    at::Tensor actual = at::empty_like(input_tensor).contiguous();
    graph.maybe_cast_and_copy_from_staging(
        staging_output, actual.mutable_data_ptr(), actual.numel(), vkapi::kInt);
    const at::Tensor expected = at::remainder(input_tensor, divisor);
    if (!at::equal(expected, actual)) {
      std::cout << "divisor " << divisor << "\nexpected " << expected
                << "\nactual " << actual << std::endl;
    }
    ASSERT_TRUE(at::equal(expected, actual));
  }
}

} // namespace

TEST(VulkanRemainderScalarTest, buffer_positive_divisor) {
  test_vulkan_remainder_scalar(vkcompute::utils::kBuffer, 128);
}

TEST(VulkanRemainderScalarTest, texture_positive_divisor) {
  test_vulkan_remainder_scalar(vkcompute::utils::kTexture3D, 128);
}

TEST(VulkanRemainderScalarTest, buffer_negative_divisor) {
  test_vulkan_remainder_scalar(vkcompute::utils::kBuffer, -7);
}

TEST(VulkanRemainderScalarTest, texture_negative_divisor) {
  test_vulkan_remainder_scalar(vkcompute::utils::kTexture3D, -7);
}
