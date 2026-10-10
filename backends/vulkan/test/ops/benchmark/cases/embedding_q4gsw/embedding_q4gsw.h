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
namespace embedding_q4gsw {

struct EmbeddingConfig {
  int64_t vocab_size;
  int64_t embed_dim;
  int64_t group_size;
  std::vector<int64_t> indices_shape;
  std::string test_case_name = "placeholder";
  vkapi::ScalarType dtype = vkapi::kHalf;
  vkapi::ScalarType scales_dtype = vkapi::kHalf;
  utils::StorageType storage_type = utils::kBuffer;
  bool is_linear_weight = false;
};

TestCase create_test_case(const EmbeddingConfig& config);

void embedding_4bit_reference(TestCase& tc);

} // namespace embedding_q4gsw
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
