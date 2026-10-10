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
namespace q8ta_unary {

constexpr int64_t kRefDimSizeLimit = 512;

struct Q8taUnaryConfig {
  std::vector<int64_t> shape;
  std::string test_case_name = "placeholder";
  std::string op_name = "q8ta_unary_test";
};

TestCase create_test_case_from_config(
    const Q8taUnaryConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout quant_layout);

void q8ta_unary_reference_impl(TestCase& test_case);

} // namespace q8ta_unary
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
