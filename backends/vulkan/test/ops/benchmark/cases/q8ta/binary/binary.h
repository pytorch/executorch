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
namespace q8ta_binary {

constexpr int64_t kRefDimSizeLimit = 512;

// Configuration struct for q8ta binary testing
struct Q8taBinaryConfig {
  std::vector<int64_t> shape; // Tensor shape (can be any dimensionality)
  std::string test_case_name = "placeholder";
  std::string op_name = "q8ta_add";
  // Shape of input B; empty means same as `shape`. Inputs are broadcast
  // against each other.
  std::vector<int64_t> other_shape = {};
};

TestCase create_test_case_from_config(
    const Q8taBinaryConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout quant_layout,
    bool const_b = false);

void q8ta_add_reference_impl(TestCase& test_case);

} // namespace q8ta_binary
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
