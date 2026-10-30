// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark/framework/conv2d_utils.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/utils.h>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d_transposed {

constexpr int64_t kRefDimSizeLimit = 100;

TestCase create_test_case_from_config(
    const Conv2dConfig& config,
    int32_t output_pad_h,
    int32_t output_pad_w,
    vkapi::ScalarType input_dtype,
    utils::StorageType fp_storage_type,
    utils::GPUMemoryLayout int8_memory_layout);

void reference_impl(TestCase& test_case);

int64_t quantized_conv2d_transposed_flop_calculator(const TestCase& test_case);

} // namespace q8ta_conv2d_transposed
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
