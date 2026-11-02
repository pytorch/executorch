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
namespace q8csw_conv2d {

constexpr int64_t kRefDimSizeLimit = 100;

TestCase create_test_case_from_config(
    const Conv2dConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype);

void reference_impl(TestCase& test_case);

int64_t quantized_conv2d_flop_calculator(const TestCase& test_case);

} // namespace q8csw_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
