// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/utils.h>

#include <string>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8csw_linear {

constexpr int64_t kRefDimSizeLimit = 300;

// Linear configuration struct
struct LinearConfig {
  int64_t M; // Batch size / number of rows in input
  int64_t K; // Input features / columns in input, rows in weight
  int64_t N; // Output features / columns in weight
  bool has_bias = true;
  std::string test_case_name = "placeholder";
  std::string op_name = "linear_q8ta_q8csw";
};

TestCase create_test_case_from_config(
    const LinearConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype);

void reference_impl(TestCase& test_case);

int64_t quantized_linear_flop_calculator(const TestCase& test_case);

} // namespace q8csw_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
