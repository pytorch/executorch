// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark/framework/conv2d_utils.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/utils.h>

#include <string>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d {

constexpr int64_t kRefDimSizeLimit = 100;
constexpr int64_t kRefOperationLimit = 2 * 1024 * 1024;

struct Im2colUnsignedTestOptions {
  int32_t input_zero_point = 2;
  int32_t output_zero_point = -1;
  bool use_extreme_values = false;
  bool use_accumulator_limit_values = false;
  bool has_bias = true;
  const char* activation = "none";
  float weight_scale = 1.0f / 256.0f;
};

struct PointwiseTestOptions {
  int32_t input_zero_point = 2;
  bool has_bias = true;
};

TestCase create_test_case_from_config(
    const Conv2dConfig& config,
    vkapi::ScalarType input_dtype,
    utils::StorageType fp_storage_type,
    utils::GPUMemoryLayout int8_memory_layout,
    const std::string& impl_selector = "",
    const Im2colUnsignedTestOptions* im2col_options = nullptr,
    const float input_scale_val = 0.008123f,
    const DataGenType input_data_gen = DataGenType::RANDOM);

TestCase create_scenex_test_case(
    const Conv2dConfig& source_config,
    const std::string& route);

TestCase create_dw_test_case_from_config(
    const Conv2dConfig& config,
    vkapi::ScalarType input_dtype,
    utils::StorageType fp_storage_type,
    utils::GPUMemoryLayout int8_memory_layout,
    const std::string& impl_selector = "");

TestCase create_pw_test_case_from_config(
    const Conv2dConfig& config,
    vkapi::ScalarType input_dtype,
    utils::StorageType fp_storage_type,
    utils::GPUMemoryLayout int8_memory_layout,
    const std::string& impl_selector = "",
    const PointwiseTestOptions& options = {});

void reference_impl(TestCase& test_case);

void scenex_direct_reference(TestCase& test_case);

void dw_reference_impl(TestCase& test_case);

void pw_reference_impl(TestCase& test_case);

int64_t quantized_conv2d_flop_calculator(const TestCase& test_case);

int64_t quantized_conv2d_dw_flop_calculator(const TestCase& test_case);

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
