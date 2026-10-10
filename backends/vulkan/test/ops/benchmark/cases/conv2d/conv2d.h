// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark/framework/conv2d_utils.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/utils.h>

#include <string>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv2d {

constexpr int64_t kRefDimSizeLimit = 64;

// Conv2d shaders are texture-only and require channels-packed layout
constexpr utils::StorageType kStorageType = utils::kTexture3D;
constexpr utils::GPUMemoryLayout kMemoryLayout = utils::kChannelsPacked;

struct InputDims {
  int64_t N;
  int64_t C;
  int64_t H;
  int64_t W;

  InputDims(int64_t n, int64_t c, int64_t h, int64_t w)
      : N(n), C(c), H(h), W(w) {}
};

struct Conv2dTestConfig {
  InputDims dims;
  int64_t C_out;
  KernelSize kernel;
  Stride stride;
  Padding padding;
  Dilation dilation;
  bool has_bias;
};

struct Conv2dDwConfig {
  InputDims dims;
  KernelSize kernel;
  Stride stride;
  Padding padding;
  Dilation dilation;
  bool has_bias;
};

struct Conv2dPwConfig {
  int64_t N;
  int64_t C_in;
  int64_t C_out;
  int64_t H;
  int64_t W;
  bool has_bias;
};

bool conv2d_is_perf_shape(int64_t C_in, int64_t C_out, int64_t H, int64_t W);

TestCase create_conv2d_test_case(
    const Conv2dTestConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout,
    const std::string& impl_selector = "");

TestCase create_conv2d_dw_test_case(
    const Conv2dDwConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout,
    const std::string& impl_selector = "");

TestCase create_conv2d_pw_test_case(
    const Conv2dPwConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout);

void conv2d_reference_impl(TestCase& test_case);

void conv2d_dw_reference_impl(TestCase& test_case);

void conv2d_pw_reference_impl(TestCase& test_case);

int64_t conv2d_flop_calculator(const TestCase& test_case);

int64_t conv2d_dw_flop_calculator(const TestCase& test_case);

int64_t conv2d_pw_flop_calculator(const TestCase& test_case);

} // namespace conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
