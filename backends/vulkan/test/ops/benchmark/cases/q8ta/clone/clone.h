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
namespace q8ta_clone {

constexpr int64_t kRefDimSizeLimit = 512;

// Configuration struct for q8ta clone testing
struct Q8taCloneConfig {
  std::vector<int64_t> shape; // Tensor shape (can be any dimensionality)
  std::string test_case_name = "placeholder";
  std::string op_name = "q8ta_clone_test";
};

TestCase create_test_case_from_config(
    const Q8taCloneConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout inp_quant_layout,
    utils::GPUMemoryLayout outp_quant_layout);

void q8ta_clone_reference_impl(TestCase& test_case);

// "PERF" if any dim of shape exceeds kRefDimSizeLimit, else "ACCU".
std::string get_prefix(const std::vector<int64_t>& shape);

} // namespace q8ta_clone
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
