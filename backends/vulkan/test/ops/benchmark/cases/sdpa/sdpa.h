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
namespace sdpa {

// Correctness is only checked for these small shapes; larger perf shapes throw
// std::invalid_argument from the reference (framework marks them SKIPPED) to
// avoid an O(H*S*context*D) CPU reference on the large perf matrix.
constexpr int64_t kRefContextLenLimit = 256;

// LLM SDPA (llama.custom_sdpa) shape:
//   q:       [1, S,           n_heads,    head_dim]  (DHSB, width-packed)
//   k/v cache:[1, context_len, n_kv_heads, head_dim]
struct SDPAConfig {
  int64_t head_dim;
  int64_t n_heads;
  int64_t n_kv_heads;
  int64_t seq_len; // S: query tokens (1 for decode, >1 for prefill)
  int64_t context_len; // total KV length (kv_len)
  std::string model; // label only
  std::string regime; // "decode" / "prefill", label only
};

TestCase create_sdpa_test_case(
    const SDPAConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    const std::string& impl);

void sdpa_reference_impl(TestCase& test_case);

int64_t sdpa_flop_calculator(const TestCase& test_case);

} // namespace sdpa
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
