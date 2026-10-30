// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/clone/clone.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_clone {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<utils::GPUMemoryLayout>& fp_layouts,
    const std::vector<utils::GPUMemoryLayout>& quant_layouts,
    const std::vector<utils::StorageType>& storage_types) {
  std::vector<TestCase> test_cases;

  // Generate all combinations
  for (const auto& shape : shapes) {
    std::string prefix = get_prefix(shape);

    for (const auto& fp_layout : fp_layouts) {
      for (const auto& inp_quant_layout : quant_layouts) {
        for (const auto& outp_quant_layout : quant_layouts) {
          for (const auto& storage_type : storage_types) {
            Q8taCloneConfig config;
            config.shape = shape;
            config.test_case_name = prefix;

            test_cases.push_back(create_test_case_from_config(
                config,
                storage_type,
                vkapi::kFloat,
                fp_layout,
                inp_quant_layout,
                outp_quant_layout));
          }
        }
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/clone", "correctness") {
  // Shapes to test
  const std::vector<std::vector<int64_t>> shapes = {
      // Small test cases for correctness
      {1, 3, 16, 16},
      {1, 8, 32, 32},
      {1, 16, 24, 24},
      {1, 32, 12, 12},
      {1, 1, 64, 64},
      {1, 3, 64, 64},
      {1, 4, 16, 16},

      // Different tensor sizes
      {1, 8, 20, 20},
      {1, 16, 14, 14},
      {1, 8, 28, 28},

      // Odd tensor sizes
      {1, 3, 15, 15},
      {1, 13, 31, 31},
      {1, 17, 23, 23},

      // Performance test cases (larger tensors)
      {1, 64, 128, 128},
      {1, 32, 64, 64},
      {1, 128, 56, 56},
      {1, 128, 128, 128},
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
  // Test with buffer storage only
  const std::vector<utils::StorageType> storage_types = {utils::kBuffer};
  return {
      generate_test_cases(shapes, fp_layouts, quant_layouts, storage_types),
      q8ta_clone_reference_impl};
}

} // namespace q8ta_clone
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
