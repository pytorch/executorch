// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <iostream>
#include <vector>

#include "utils.h"

using namespace executorch::vulkan::prototyping;

using namespace vkcompute;

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

// CPU reference: unpack 4-bit weights, dequantize, and perform embedding lookup
void embedding_4bit_reference(TestCase& tc) {
  auto& weight_spec = tc.inputs()[0];
  auto& scales_spec = tc.inputs()[1];
  int32_t group_size = tc.inputs()[2].get_int_value();
  auto& indices_spec = tc.inputs()[3];
  bool is_linear_weight = tc.inputs()[4].get_bool_value();
  auto& output_spec = tc.outputs()[0];

  weight_spec.ensure_data_generated();
  scales_spec.ensure_data_generated();
  indices_spec.ensure_data_generated();

  const auto& weight_data = weight_spec.get_uint8_data();
  const auto& indices_data = indices_spec.get_int32_data();

  bool scales_are_half = (scales_spec.dtype == vkapi::kHalf);

  int64_t packed_dim = weight_spec.sizes[1];
  int64_t embed_dim = packed_dim * 2;
  int64_t groups_per_row = scales_spec.sizes[1];

  int64_t num_indices = 1;
  for (auto s : indices_spec.sizes) {
    num_indices *= s;
  }

  int64_t total_output = num_indices * embed_dim;

  // Always populate ref_float_data so the caching framework can distribute it
  output_spec.get_ref_float_data().resize(total_output);

  bool output_is_half = (output_spec.dtype == vkapi::kHalf);
  if (output_is_half) {
    output_spec.get_ref_half_data().resize(total_output);
  }

  for (int64_t i = 0; i < num_indices; ++i) {
    int32_t idx = indices_data[i];
    for (int64_t d = 0; d < embed_dim; ++d) {
      int64_t packed_idx = d / 2;
      uint8_t packed_byte = weight_data[idx * packed_dim + packed_idx];

      // Unpack: packed_byte = (even_val + 8) << 4 | (odd_val + 8)
      // Even d -> high nibble, odd d -> low nibble
      // For linear weight packing, nibble order is swapped
      int int4_val;
      if (d % 2 == 0) {
        if (is_linear_weight) {
          int4_val = static_cast<int>(packed_byte & 0xF) - 8;
        } else {
          int4_val = static_cast<int>(packed_byte >> 4) - 8;
        }
      } else {
        if (is_linear_weight) {
          int4_val = static_cast<int>(packed_byte >> 4) - 8;
        } else {
          int4_val = static_cast<int>(packed_byte & 0xF) - 8;
        }
      }

      int64_t group_idx = d / group_size;
      int64_t scale_idx = idx * groups_per_row + group_idx;

      float scale;
      if (scales_are_half) {
        uint16_t scale_half = scales_spec.get_half_data()[scale_idx];
        scale = half_to_float(scale_half);
      } else {
        scale = scales_spec.get_float_data()[scale_idx];
      }

      float result = static_cast<float>(int4_val) * scale;

      // Always store float reference
      output_spec.get_ref_float_data()[i * embed_dim + d] = result;

      if (output_is_half) {
        output_spec.get_ref_half_data()[i * embed_dim + d] =
            float_to_half(result);
      }
    }
  }
}

TestCase create_test_case(const EmbeddingConfig& config) {
  TestCase test_case;
  test_case.set_name(config.test_case_name);
  test_case.set_operator_name("et_vk.embedding_q4gsw.default");
  test_case.set_shader_filter({});

  // Weight: [vocab_size, embed_dim / 2] packed uint8
  ValueSpec weight(
      {config.vocab_size, config.embed_dim / 2},
      vkapi::kByte,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::RANDINT4);
  weight.set_constant(true);
  test_case.add_input_spec(weight);

  // Weight scales: [vocab_size, groups_per_row]
  int64_t groups_per_row = config.embed_dim / config.group_size;
  ValueSpec weight_scales(
      {config.vocab_size, groups_per_row},
      config.scales_dtype,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);
  test_case.add_input_spec(weight_scales);

  // Group size: int scalar
  ValueSpec group_size_spec(static_cast<int32_t>(config.group_size));
  test_case.add_input_spec(group_size_spec);

  // Indices: [batch, seq_len] int32
  ValueSpec indices(
      config.indices_shape,
      vkapi::kInt,
      config.storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT);

  // Clamp indices to valid vocab range
  indices.ensure_data_generated();
  for (auto& idx : indices.get_int32_data()) {
    idx = std::abs(idx) % config.vocab_size;
  }

  test_case.add_input_spec(indices);

  // is_linear_weight: bool scalar
  ValueSpec is_linear_weight_spec(config.is_linear_weight);
  test_case.add_input_spec(is_linear_weight_spec);

  // Output: indices.shape + [embed_dim]
  std::vector<int64_t> output_shape = config.indices_shape;
  output_shape.push_back(config.embed_dim);
  ValueSpec output(
      output_shape, config.dtype, config.storage_type, utils::kWidthPacked);
  test_case.add_output_spec(output);

  return test_case;
}

std::vector<TestCase> generate_test_cases() {
  std::vector<TestCase> test_cases;

  // --- is_linear_weight = true ---

  // Basic test with linear weight packing
  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_linear_weight",
       .is_linear_weight = true}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_linear_weight",
       .is_linear_weight = true}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_linear_weight",
       .is_linear_weight = true}));

  // fp32 output with linear weight
  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32_linear_weight",
       .dtype = vkapi::kFloat,
       .is_linear_weight = true}));

  // Texture3D with linear weight
  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_texture_linear_weight",
       .storage_type = utils::kTexture3D,
       .is_linear_weight = true}));

  // Llama 3.2 1B with linear weight packing
  test_cases.push_back(create_test_case(
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp32_linear_weight",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kBuffer,
       .is_linear_weight = true}));

  // --- Half scales (default) ---

  // Basic test: small vocab, small embed_dim
  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d"}));

  // 2D indices
  test_cases.push_back(create_test_case(
      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d"}));

  // Larger vocab, multiple groups
  test_cases.push_back(create_test_case(
      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup"}));

  // group_size == embed_dim (single group)
  test_cases.push_back(create_test_case(
      {.vocab_size = 50,
       .embed_dim = 64,
       .group_size = 64,
       .indices_shape = {2, 4},
       .test_case_name = "single_group"}));

  // fp32 output variants (half scales)
  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32",
       .dtype = vkapi::kFloat}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_fp32",
       .dtype = vkapi::kFloat}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_fp32",
       .dtype = vkapi::kFloat}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 50,
       .embed_dim = 64,
       .group_size = 64,
       .indices_shape = {2, 4},
       .test_case_name = "single_group_fp32",
       .dtype = vkapi::kFloat}));

  // Texture3D variants (fp16 output, half scales)
  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_texture",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kTexture3D}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_texture",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kTexture3D}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_texture",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kTexture3D}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 50,
       .embed_dim = 64,
       .group_size = 64,
       .indices_shape = {2, 4},
       .test_case_name = "single_group_texture",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kTexture3D}));

  // Texture3D variants (fp32 output, half scales)
  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32_texture",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_fp32_texture",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_fp32_texture",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 50,
       .embed_dim = 64,
       .group_size = 64,
       .indices_shape = {2, 4},
       .test_case_name = "single_group_fp32_texture",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D}));

  // --- Float scales ---

  // Buffer variants with float scales
  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_float_scales",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_float_scales",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_float_scales",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32_float_scales",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 32,
       .embed_dim = 64,
       .group_size = 32,
       .indices_shape = {2, 3},
       .test_case_name = "small_2d_fp32_float_scales",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 100,
       .embed_dim = 128,
       .group_size = 32,
       .indices_shape = {4, 8},
       .test_case_name = "medium_multigroup_fp32_float_scales",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat}));

  // Texture3D with float scales
  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_float_scales_texture",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D}));

  test_cases.push_back(create_test_case(
      {.vocab_size = 16,
       .embed_dim = 32,
       .group_size = 32,
       .indices_shape = {4},
       .test_case_name = "small_1d_fp32_float_scales_texture",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat,
       .storage_type = utils::kTexture3D}));

  // Llama 3.2 1B prefill configuration (fp32 output, half scales)
  test_cases.push_back(create_test_case(
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp32",
       .dtype = vkapi::kFloat,
       .storage_type = utils::kBuffer}));

  // Llama 3.2 1B prefill configuration (fp32 output, float scales)
  test_cases.push_back(create_test_case(
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp32_float_scales",
       .dtype = vkapi::kFloat,
       .scales_dtype = vkapi::kFloat,
       .storage_type = utils::kBuffer}));

  // Llama 3.2 1B prefill configuration (fp16 output, half scales)
  test_cases.push_back(create_test_case(
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp16",
       .dtype = vkapi::kHalf,
       .storage_type = utils::kBuffer}));

  // Llama 3.2 1B prefill configuration (fp16 output, float scales)
  test_cases.push_back(create_test_case(
      {.vocab_size = 128256,
       .embed_dim = 2048,
       .group_size = 32,
       .indices_shape = {1, 2047},
       .test_case_name = "llama_3_2_1b_prefill_fp16_float_scales",
       .dtype = vkapi::kHalf,
       .scales_dtype = vkapi::kFloat,
       .storage_type = utils::kBuffer}));

  return test_cases;
}

int main(int argc, char** argv) {
  auto results = execute_test_cases(
      generate_test_cases,
      "embedding_q4gsw",
      /*warmup_runs = */ 1,
      /*benchmark_runs = */ 1,
      embedding_4bit_reference);

  results.print_summary();

  if (results.get_failed_count() > 0) {
    std::cerr << "FAILED: " << results.get_failed_count() << " test(s) failed."
              << std::endl;
    return 1;
  }

  std::cout << "PASSED: All tests passed." << std::endl;
  return 0;
}
