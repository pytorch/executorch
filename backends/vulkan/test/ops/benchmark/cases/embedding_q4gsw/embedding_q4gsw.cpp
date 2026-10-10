// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/embedding_q4gsw/embedding_q4gsw.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace embedding_q4gsw {

TestCase create_test_case(const EmbeddingConfig& config) {
  TestCase test_case;

  // Compute the output shape for label: indices_shape + [embed_dim]
  std::vector<int64_t> output_shape_for_label = config.indices_shape;
  output_shape_for_label.push_back(config.embed_dim);

  // Treat any case that uses the large llama vocab/embed sizes as PERF.
  // Otherwise default to ACCU (no PERF distinction is exercised by this op).
  bool is_perf = config.vocab_size > 1024 || config.embed_dim > 1024;
  std::string prefix = is_perf ? "PERF" : "ACCU";

  std::string in_dtype = dtype_short(vkapi::kInt); // indices are i32
  std::string out_dtype = dtype_short(config.dtype);

  std::string storage_str = repr_str(config.storage_type, utils::kWidthPacked);

  std::string shape_str = shape_bracket(config.indices_shape) + "x[" +
      std::to_string(config.vocab_size) + "," +
      std::to_string(config.embed_dim) + "]";
  shape_str += " g" + std::to_string(config.group_size);
  if (config.is_linear_weight) {
    shape_str += " lw";
  }
  std::string suffix;
  if (config.scales_dtype == vkapi::kFloat) {
    suffix = "[f32_scales]";
  }
  std::string name = make_test_label(
      prefix, in_dtype, out_dtype, shape_str, storage_str, suffix);
  test_case.set_name(name);
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

} // namespace embedding_q4gsw
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
