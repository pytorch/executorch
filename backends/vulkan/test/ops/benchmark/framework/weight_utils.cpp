// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "weight_utils.h"
#include "config.h"

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

ValueRef quantized_weights_canvas(
    ComputeGraph& graph,
    const ValueRef weight_ref) {
  const auto original_sizes = graph.sizes_of(weight_ref);

  std::vector<int64_t> sorted_sizes = original_sizes;
  std::sort(sorted_sizes.begin(), sorted_sizes.end(), std::greater<int64_t>());
  int64_t largest1 = sorted_sizes.size() > 0 ? sorted_sizes[0] : 0;

  std::vector<int64_t> final_sizes = {1, largest1, largest1};

  // Debug logging if debugging flag is set
  if (debugging()) {
    std::cout << "Debug: Creating quantized weights canvas tensor" << std::endl;
    std::cout << "Debug: Original sizes: [";
    for (size_t i = 0; i < original_sizes.size(); ++i) {
      std::cout << original_sizes[i];
      if (i < original_sizes.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Debug: Canvas sizes: [";
    for (size_t i = 0; i < final_sizes.size(); ++i) {
      std::cout << final_sizes[i];
      if (i < final_sizes.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  ValueRef packed_weight = graph.add_tensor(
      final_sizes, vkapi::kInt, utils::kTexture3D, utils::kWidthPacked);

  std::string kernel_name = "packed_int32_canvas";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_gwg(packed_weight),
      graph.create_lwg(packed_weight),
      weight_ref,
      packed_weight,
      // UBOs
      {graph.logical_limits_ubo(packed_weight)},
      // Specialization constants
      {},
      // Push Constants
      {}));

  return packed_weight;
}

ValueRef float_tensor_canvas(ComputeGraph& graph, const ValueRef weight_ref) {
  const auto original_sizes = graph.sizes_of(weight_ref);

  std::vector<int64_t> sorted_sizes = original_sizes;
  std::sort(sorted_sizes.begin(), sorted_sizes.end(), std::greater<int64_t>());
  int64_t largest1 = sorted_sizes.size() > 0 ? sorted_sizes[0] : 0;

  std::vector<int64_t> final_sizes = {1, largest1, largest1};

  // Debug logging if debugging flag is set
  if (debugging()) {
    std::cout << "Debug: Creating float tensor canvas" << std::endl;
    std::cout << "Debug: Original sizes: [";
    for (size_t i = 0; i < original_sizes.size(); ++i) {
      std::cout << original_sizes[i];
      if (i < original_sizes.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Debug: Canvas sizes: [";
    for (size_t i = 0; i < final_sizes.size(); ++i) {
      std::cout << final_sizes[i];
      if (i < final_sizes.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  ValueRef packed_weight = graph.add_tensor(
      final_sizes, vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR("float_canvas"),
      graph.create_gwg(packed_weight),
      graph.create_lwg(packed_weight),
      weight_ref,
      packed_weight,
      // UBOs
      {graph.logical_limits_ubo(packed_weight)},
      // Specialization constants
      {},
      // Push Constants
      {}));

  return packed_weight;
}

// Compute weight sums for quantized operations (linear and convolution)
void compute_weight_sums(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t out_features,
    int64_t elements_per_output_feature) {
  auto& weight_sums_data = weight_sums.get_int32_data();
  auto& quantized_weight_data = quantized_weight.get_int8_data();

  // Don't resize down - the buffer may be pre-allocated with aligned size.
  // Only resize up if needed.
  if (weight_sums_data.size() < static_cast<size_t>(out_features)) {
    weight_sums_data.resize(out_features);
  }

  // For each output feature, compute the sum of quantized weights
  for (int64_t out_f = 0; out_f < out_features; ++out_f) {
    int32_t sum = 0;
    for (int64_t elem = 0; elem < elements_per_output_feature; ++elem) {
      // Weight indexing depends on the layout:
      // For linear: [out_features, in_features] -> out_f *
      // elements_per_output_feature + elem For conv2d: [C_out, C_in * K_h *
      // K_w] -> out_f * elements_per_output_feature + elem
      int64_t weight_idx = out_f * elements_per_output_feature + elem;
      sum += static_cast<int32_t>(quantized_weight_data[weight_idx]);
    }
    weight_sums_data[out_f] = sum;
  }
}

// Compute weight sums for 4D quantized conv2d operations
// Weight layout: [C_out, K_h, K_w, align_up_4(C_in_per_group)]
void compute_weight_sums_4d(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t aligned_in_channels) {
  auto& weight_sums_data = weight_sums.get_int32_data();
  auto& quantized_weight_data = quantized_weight.get_int8_data();

  weight_sums_data.resize(out_channels);

  // For each output channel, compute the sum of quantized weights
  for (int64_t out_c = 0; out_c < out_channels; ++out_c) {
    int32_t sum = 0;

    for (int64_t kh = 0; kh < kernel_h; ++kh) {
      for (int64_t kw = 0; kw < kernel_w; ++kw) {
        for (int64_t in_c = 0; in_c < aligned_in_channels; ++in_c) {
          // Weight indexing: [out_c, kh, kw, in_c]
          int64_t weight_idx =
              out_c * (kernel_h * kernel_w * aligned_in_channels) +
              kh * (kernel_w * aligned_in_channels) + kw * aligned_in_channels +
              in_c;
          sum += static_cast<int32_t>(quantized_weight_data[weight_idx]);
        }
      }
    }

    weight_sums_data[out_c] = sum;
  }
}

// Helper function to unpack 4-bit values from uint8 (same as in
// q4gsw_linear.cpp)
std::pair<int8_t, int8_t> unpack_4bit_utils(uint8_t packed) {
  // Extract lower 4 bits and upper 4 bits
  int8_t lower = packed & 0x0F;
  int8_t upper = (packed >> 4) & 0x0F;

  // Subtract 8 from unpacked 4-bit values
  lower -= 8;
  upper -= 8;

  return std::make_pair(lower, upper);
}

// Compute weight sums for 4-bit group symmetric quantized weights
void compute_weight_sums_4bit_grouped(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t num_groups,
    int64_t out_features,
    int64_t group_size) {
  auto& weight_sums_data = weight_sums.get_int32_data();
  auto& quantized_weight_data = quantized_weight.get_uint8_data();

  // Resize to [num_groups, out_features]
  weight_sums_data.resize(num_groups * out_features);

  // For each group and each output feature, compute the sum of quantized
  // weights in that group
  for (int64_t group_idx = 0; group_idx < num_groups; ++group_idx) {
    for (int64_t out_f = 0; out_f < out_features; ++out_f) {
      int32_t sum = 0;

      // Sum weights for this group and output feature
      for (int64_t in_group = 0; in_group < group_size; ++in_group) {
        int64_t in_f = group_idx * group_size + in_group;

        // Get packed weight value - weight matrix is [N, K/2]
        int64_t weight_idx =
            out_f * ((num_groups * group_size) / 2) + (in_f / 2);
        uint8_t packed_weight = quantized_weight_data[weight_idx];

        // Unpack 4-bit weight
        auto unpacked = unpack_4bit_utils(packed_weight);
        int8_t weight_4bit = (in_f % 2 == 0) ? unpacked.first : unpacked.second;

        sum += static_cast<int32_t>(weight_4bit);
      }

      // Store sum for this group and output feature
      int64_t sums_idx = group_idx * out_features + out_f;
      weight_sums_data[sums_idx] = sum;
    }
  }
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
