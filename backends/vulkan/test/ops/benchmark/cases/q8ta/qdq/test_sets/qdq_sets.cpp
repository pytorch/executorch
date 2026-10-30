// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/qdq/qdq.h>

#include <algorithm>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_qdq {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<utils::GPUMemoryLayout>& fp_layouts,
    const std::vector<utils::GPUMemoryLayout>& quant_layouts,
    const std::vector<utils::StorageType>& storage_types,
    const std::vector<utils::GPUMemoryLayout>& legacy_4w4c_fp_layouts) {
  std::vector<TestCase> test_cases;

  // Generate all combinations
  for (const auto& shape : shapes) {
    // Generate test case name prefix from shape dimensions
    std::string prefix = "ACCU";
    for (const auto& dim : shape) {
      if (dim > kRefDimSizeLimit) {
        prefix = "PERF";
        break;
      }
    }

    const bool is_highdim = shape.size() > 4;

    for (const auto& fp_layout : fp_layouts) {
      const bool test_legacy_4w4c =
          std::find(
              legacy_4w4c_fp_layouts.begin(),
              legacy_4w4c_fp_layouts.end(),
              fp_layout) != legacy_4w4c_fp_layouts.end();
      for (const auto& quant_layout : quant_layouts) {
        for (const auto& storage_type : storage_types) {
          // Textures are limited to 4D
          if (is_highdim && storage_type != utils::kBuffer) {
            continue;
          }
          QDQ8BitConfig config;
          config.shape = shape;
          config.test_case_name = prefix;

          test_cases.push_back(create_test_case_from_config(
              config, storage_type, vkapi::kFloat, fp_layout, quant_layout));
          // For 4W4C layout, also test with legacy implementation
          // (legacy path doesn't support high-dim tensors)
          if (!is_highdim && test_legacy_4w4c &&
              quant_layout == utils::kPackedInt8_4W4C) {
            test_cases.push_back(create_test_case_from_config(
                config,
                storage_type,
                vkapi::kFloat,
                fp_layout,
                quant_layout,
                /*impl_selector=*/"legacy_4w4c"));
          }
        }
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/qdq", "correctness") {
  // Shapes to test (no layout specified - will be combined with all
  // layouts)
  const std::vector<std::vector<int64_t>> shapes = {
      // 1D tensors
      {144},
      {90},

      // 2D tensors (exercises block config with ndim < 4)
      {1, 144},
      {1, 90},
      {1, 4},
      {3, 32},

      // 3D tensors
      {1, 16, 32},
      {1, 3, 64},

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

      // 5D tensors (high-dim batch support)
      {2, 1, 3, 16, 16},
      {1, 2, 8, 8, 8},

      // 6D tensors (high-dim batch support)
      {2, 1, 1, 4, 8, 16},
      {1, 1, 2, 3, 8, 8},
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
  // Test with buffer storage only - the unified block-based shaders only
  // support buffer-backed floating-point tensors. Texture storage is tested
  // separately by qdq8ta_conv2d_activations which uses the layout-specific
  // shaders.
  const std::vector<utils::StorageType> storage_types = {
      utils::kBuffer, utils::kTexture3D};
  const std::vector<utils::GPUMemoryLayout> legacy_4w4c_fp_layouts = {
      utils::kChannelsPacked};
  return {
      generate_test_cases(
          shapes,
          fp_layouts,
          quant_layouts,
          storage_types,
          legacy_4w4c_fp_layouts),
      q_dq_8bit_reference_impl};
}

// Easy test cases for q_dq_8bit operation (for debugging)
REGISTER_TEST_CASE_SET("q8ta/qdq", "debug") {
  const std::vector<std::vector<int64_t>> shapes = {
      {1, 16, 16, 16}, // 4D: [N, C, H, W]
      {1, 144}, // 2D: exercises block config with ndim < 4
      {1, 90}, // 2D: matches skin_seg model's keypoint/bbox tensor sizes
      {2, 1, 3, 16, 16}, // 5D: exercises high-dim batch support
      {2, 1, 1, 4, 8, 16}, // 6D: exercises high-dim batch support
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
  const std::vector<utils::GPUMemoryLayout> legacy_4w4c_fp_layouts = {
      utils::kWidthPacked,
      utils::kChannelsPacked,
  };
  return {
      generate_test_cases(
          shapes,
          fp_layouts,
          quant_layouts,
          storage_types,
          legacy_4w4c_fp_layouts),
      q_dq_8bit_reference_impl};
}

} // namespace q8ta_qdq
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
