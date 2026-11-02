// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/binary/binary.h>

#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_binary {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>&
        broadcast_shapes,
    const std::vector<utils::GPUMemoryLayout>& quant_layouts) {
  std::vector<TestCase> test_cases;

  std::vector<Q8taBinaryConfig> configs;
  for (const auto& shape : shapes) {
    // Generate test case name prefix from shape dimensions
    std::string prefix = "ACCU";
    for (const auto& dim : shape) {
      if (dim > kRefDimSizeLimit) {
        prefix = "PERF";
        break;
      }
    }

    Q8taBinaryConfig config;
    config.shape = shape;
    config.test_case_name = prefix;
    configs.push_back(config);
  }
  for (const auto& [shape, other_shape] : broadcast_shapes) {
    Q8taBinaryConfig config;
    config.shape = shape;
    config.test_case_name = "ACCU";
    config.other_shape = other_shape;
    configs.push_back(config);
  }

  // Generate all combinations
  for (const auto& config : configs) {
    for (const auto& quant_layout : quant_layouts) {
      test_cases.push_back(create_test_case_from_config(
          config,
          /*storage_type=*/utils::kBuffer,
          /*input_dtype=*/vkapi::kFloat,
          /*fp_memory_layout=*/utils::kWidthPacked,
          quant_layout));
      test_cases.push_back(create_test_case_from_config(
          config,
          /*storage_type=*/utils::kBuffer,
          /*input_dtype=*/vkapi::kFloat,
          /*fp_memory_layout=*/utils::kWidthPacked,
          quant_layout,
          /*const_b=*/true));
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/binary", "correctness") {
  // Shapes to test
  const std::vector<std::vector<int64_t>> shapes = {
      // 1D tensors
      {144},
      {90},

      // 3D tensors
      {1, 16, 32},
      {1, 3, 64},

      // 2D tensors (exercises block config with ndim < 4)
      {1, 144},
      {1, 90},
      {1, 4},
      {3, 32},

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
  // Quantized memory layouts to test
  const std::vector<utils::GPUMemoryLayout> quant_layouts = {
      utils::kPackedInt8_4W,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4H4W,
      utils::kPackedInt8_4C1W,
  };
  // Broadcast cases: {input A shape, input B shape}
  const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      broadcast_shapes = {
          // Per-video context row added to every frame row
          {{1, 60, 512}, {1, 1, 512}},
          {{1, 1, 512}, {1, 60, 512}},
          {{1, 16, 32}, {1, 16, 1}},
          {{1, 16, 32}, {32}},
          {{1, 8, 16, 16}, {1, 8, 1, 1}},
          {{1, 8, 16, 16}, {1, 1, 16, 16}},
          {{2, 8, 6, 6}, {1, 8, 6, 6}},
          {{1, 13, 7, 9}, {1, 1, 7, 1}},
      };
  return {
      generate_test_cases(shapes, broadcast_shapes, quant_layouts),
      q8ta_add_reference_impl};
}

// Easy test cases for q8ta_add operation (for debugging)
REGISTER_TEST_CASE_SET("q8ta/binary", "debug") {
  const std::vector<std::vector<int64_t>> shapes = {
      {1, 16, 16, 16}, // 4D: [N, C, H, W]
      {1, 144}, // 2D: exercises block config with ndim < 4
      {1, 90}, // 2D: matches skin_seg model's keypoint/bbox tensor sizes
  };
  // Quantized memory layouts to test
  const std::vector<utils::GPUMemoryLayout> quant_layouts = {
      utils::kPackedInt8_4W,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4H4W,
      utils::kPackedInt8_4C1W,
  };
  const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      broadcast_shapes = {};
  return {
      generate_test_cases(shapes, broadcast_shapes, quant_layouts),
      q8ta_add_reference_impl};
}

} // namespace q8ta_binary
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
