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
namespace q8ta_linear {

constexpr int64_t kRefDimSizeLimit = 300;

struct LinearConfig {
  int64_t M;
  int64_t K;
  int64_t N;
  bool has_bias = true;
  std::string test_case_name = "placeholder";
};

TestCase create_test_case_from_config(
    const LinearConfig& config,
    vkapi::ScalarType input_dtype,
    const std::string& impl_selector = "");

void reference_impl(TestCase& test_case);

int64_t q8ta_linear_flop_calculator(const TestCase& test_case);

} // namespace q8ta_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
