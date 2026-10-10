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
namespace choose_qparams_per_row {

constexpr int64_t kRefDimSizeLimit = 2050;

// ChooseQParams configuration struct
struct ChooseQParamsConfig {
  int64_t num_channels; // Height dimension (number of channels)
  int64_t channel_size; // Width dimension (size per channel)
  int32_t quant_min = -128;
  int32_t quant_max = 127;
  std::string test_case_name = "placeholder";
  std::string op_name = "choose_qparams_per_row";
};

TestCase create_test_case_from_config(
    const ChooseQParamsConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype);

void reference_impl(TestCase& test_case);

int64_t choose_qparams_per_channel_flop_calculator(const TestCase& test_case);

} // namespace choose_qparams_per_row
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
