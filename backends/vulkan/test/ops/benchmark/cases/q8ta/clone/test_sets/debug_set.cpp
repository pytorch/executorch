// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/clone/clone.h>

#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_clone {

namespace {

std::vector<TestCase> generate_same_layout_test_cases(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<utils::GPUMemoryLayout>& fp_layouts,
    const std::vector<utils::GPUMemoryLayout>& quant_layouts,
    const std::vector<utils::StorageType>& storage_types,
    const std::vector<vkapi::ScalarType>& float_types) {
  std::vector<TestCase> test_cases;

  for (const auto& shape : shapes) {
    Q8taCloneConfig config = {shape, get_prefix(shape)};

    // Generate test cases for each combination (same layout for input and
    // output)
    for (const auto& fp_layout : fp_layouts) {
      for (const auto& quant_layout : quant_layouts) {
        for (const auto& storage_type : storage_types) {
          for (const auto& input_dtype : float_types) {
            // Same layout: should be a simple copy
            test_cases.push_back(create_test_case_from_config(
                config,
                storage_type,
                input_dtype,
                fp_layout,
                quant_layout,
                quant_layout));
          }
        }
      }
    }
  }

  return test_cases;
}

} // namespace

// Easy test cases for q8ta_clone operation (for debugging)
REGISTER_TEST_CASE_SET("q8ta/clone", "debug") {
  // Single simple configuration for debugging
  const std::vector<std::vector<int64_t>> shapes = {
      {1, 16, 16, 16}, // shape: [N, C, H, W]
  };
  // FP memory layouts to test
  const std::vector<utils::GPUMemoryLayout> fp_layouts = {
      utils::kWidthPacked,
      utils::kChannelsPacked,
  };
  // Quantized memory layouts to test
  const std::vector<utils::GPUMemoryLayout> quant_layouts = {
      utils::kPackedInt8_4W,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4H4W,
      utils::kPackedInt8_4C1W,
  };
  const std::vector<utils::StorageType> storage_types = {utils::kBuffer};
  const std::vector<vkapi::ScalarType> float_types = {vkapi::kFloat};
  return {
      generate_same_layout_test_cases(
          shapes, fp_layouts, quant_layouts, storage_types, float_types),
      q8ta_clone_reference_impl};
}

} // namespace q8ta_clone
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
