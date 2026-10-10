// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/embedding_q4gsw/embedding_q4gsw.h>

#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace embedding_q4gsw {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<EmbeddingConfig>& configs) {
  std::vector<TestCase> test_cases;
  test_cases.reserve(configs.size());
  for (const auto& config : configs) {
    test_cases.push_back(create_test_case(config));
  }
  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("embedding_q4gsw", "correctness") {
  const std::vector<EmbeddingConfig> configs = {
      // --- is_linear_weight = true ---

      // Basic test with linear weight packing
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_linear_weight",
       .is_linear_weight = true},

      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_linear_weight",
       .is_linear_weight = true},

      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_linear_weight",
       .is_linear_weight = true},

      // fp32 output with linear weight
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32_linear_weight",
       .dtype = vkapi::kFloat,
       .is_linear_weight = true},

      // Texture3D with linear weight
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_texture_linear_weight",
       .storage_type = utils::kTexture3D,
       .is_linear_weight = true},

      // --- Half scales (default) ---

      // Basic test: small vocab, small embed_dim
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d"},

      // 2D indices
      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d"},

      // Larger vocab, multiple groups
      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup"},

      // group_size == embed_dim (single group)
      {.vocab_size = 50,
       .embed_dim = 64,
       .group_size = 64,
       .indices_shape = {2, 4},
       .test_case_name = "single_group"},

      // fp32 output variants (half scales)
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32",
       .dtype = vkapi::kFloat},

      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_fp32",
       .dtype = vkapi::kFloat},

      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_fp32",
       .dtype = vkapi::kFloat},

      {.vocab_size = 50,
       .embed_dim = 64,
       .group_size = 64,
       .indices_shape = {2, 4},
       .test_case_name = "single_group_fp32",
       .dtype = vkapi::kFloat},

      // Texture3D variants (fp16 output, half scales)
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_texture",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kTexture3D},

      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_texture",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kTexture3D},

      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_texture",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kTexture3D},

      {.vocab_size = 50,
       .embed_dim = 64,
       .group_size = 64,
       .indices_shape = {2, 4},
       .test_case_name = "single_group_texture",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kTexture3D},

      // Texture3D variants (fp32 output, half scales)
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32_texture",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D},

      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_fp32_texture",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D},

      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_fp32_texture",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D},

      {.vocab_size = 50,
       .embed_dim = 64,
       .group_size = 64,
       .indices_shape = {2, 4},
       .test_case_name = "single_group_fp32_texture",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D},

      // --- Float scales ---

      // Buffer variants with float scales
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_float_scales",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat},

      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_float_scales",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat},

      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_float_scales",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat},

      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32_float_scales",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat},

      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_fp32_float_scales",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat},

      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_fp32_float_scales",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat},

      // Texture3D with float scales
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_float_scales_texture",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D},

      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32_float_scales_texture",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D},
  };
  return {generate_test_cases(configs), embedding_4bit_reference};
}

REGISTER_TEST_CASE_SET("embedding_q4gsw", "performance") {
  const std::vector<EmbeddingConfig> configs = {
      // Llama 3.2 1B with linear weight packing
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp32_linear_weight",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kBuffer,
       .is_linear_weight = true},

      // Llama 3.2 1B prefill configuration (fp32 output, half scales)
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp32",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kBuffer},

      // Llama 3.2 1B prefill configuration (fp32 output, float scales)
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp32_float_scales",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat,
       .storage_type = utils::kBuffer},

      // Llama 3.2 1B prefill configuration (fp16 output, half scales)
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp16",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kBuffer},

      // Llama 3.2 1B prefill configuration (fp16 output, float scales)
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp16_float_scales",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat,
       .storage_type = utils::kBuffer},
  };
  return {generate_test_cases(configs), embedding_4bit_reference};
}

} // namespace embedding_q4gsw
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
