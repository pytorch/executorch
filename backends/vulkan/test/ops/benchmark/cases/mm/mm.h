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
namespace mm {

constexpr int64_t kRefDimSizeLimit = 256;

struct MmConfig {
  // mat1: [M, K] or [B, M, K]
  // mat2: [K, N] or [B, K, N]
  int64_t B; // batch size, 0 for non-batched
  int64_t M;
  int64_t K;
  int64_t N;
  bool has_bias; // true for addmm/linear
  bool mat2_is_transposed; // true for linear (weight is [N, K])
  bool mat2_is_constant; // true to test prepacked linear path
  // "default" routes through aten.{mm,linear,...}.default (production path).
  // "coopmat" / "tiled" force-dispatch a specific shader implementation.
  std::string impl_selector = "default";
};

TestCase create_mm_test_case(
    const MmConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout);

void reference_impl(TestCase& test_case);

int64_t mm_flop_calculator(const TestCase& test_case);

} // namespace mm
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
