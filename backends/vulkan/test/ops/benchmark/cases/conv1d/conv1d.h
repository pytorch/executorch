// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/utils.h>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv1d {

constexpr int64_t kRefDimSizeLimit = 256;

struct Conv1dDWConfig {
  int64_t N;
  int64_t C;
  int64_t L;
  int64_t K;
  int64_t stride;
  int64_t padding;
  int64_t dilation;
  bool has_bias;
};

struct Conv1dPWConfig {
  int64_t N;
  int64_t C_in;
  int64_t C_out;
  int64_t L;
  bool has_bias;
};

TestCase create_conv1d_dw_test_case(
    const Conv1dDWConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type);

TestCase create_conv1d_pw_test_case(
    const Conv1dPWConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type);

void conv1d_dw_reference_impl(TestCase& test_case);

void conv1d_pw_reference_impl(TestCase& test_case);

int64_t conv1d_dw_flop_calculator(const TestCase& test_case);

int64_t conv1d_pw_flop_calculator(const TestCase& test_case);

} // namespace conv1d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
