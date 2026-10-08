// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/utils.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_qdq {

constexpr int64_t kRefDimSizeLimit = 512;

// Configuration struct for tensor quantize-dequantize testing
struct QDQ8BitConfig {
  std::vector<int64_t> shape; // Tensor shape (can be any dimensionality)
  std::string test_case_name = "placeholder";
  std::string op_name = "q_dq_8bit_per_tensor";
};

TestCase create_test_case_from_config(
    const QDQ8BitConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout quantized_memory_layout,
    const std::string& impl_selector = "");

void q_dq_8bit_reference_impl(TestCase& test_case);

} // namespace q8ta_qdq
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
