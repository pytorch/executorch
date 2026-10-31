// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>
#include <algorithm>
#include <cmath>
#include <utility>

namespace vkcompute::shader_profiling::reference::q8ta {
inline float scale(const Json& spec, size_t i) {
  return spec.at("args").at(i).at("float").get<float>();
}
inline int64_t integer(const Json& spec, size_t i) {
  return spec.at("args").at(i).at("int").get<int64_t>();
}
inline int64_t requantize(float value, float inverse_scale, int64_t zp) {
  const float scaled = value * inverse_scale;
  if (scaled < float(-128 - zp) - 1.0f)
    return -128;
  if (scaled > float(127 - zp) + 1.0f)
    return 127;
  const float base = std::floor(scaled);
  const float fraction = scaled - base;
  const bool round_up =
      fraction > .5f || (fraction == .5f && std::fmod(base, 2.0f) != 0.0f);
  const int64_t rounded = int64_t(base) + int64_t(round_up);
  return std::clamp(rounded + zp, int64_t(-128), int64_t(127));
}
inline int64_t quantize(float value, float s, int64_t zp) {
  return requantize(value, 1.0f / s, zp);
}
inline int64_t requantize_accumulator(
    int64_t corrected_accumulator,
    float input_scale,
    float weight_scale,
    float output_scale,
    int64_t output_zero_point,
    float bias = 0.0f,
    bool relu = false,
    bool fused_bias = true) {
  const float combined_scale = weight_scale * input_scale;
  float value = std::fma(
      float(corrected_accumulator), combined_scale, fused_bias ? bias : 0.0f);
  if (!fused_bias)
    value += bias;
  if (relu)
    value = std::max(value, 0.0f);
  return requantize(value, 1.0f / output_scale, output_zero_point);
}
inline float dequantize(int64_t value, float s, int64_t zp) {
  return float(value - zp) * s;
}
inline Tensor quantized_input(const Tensor& input, float s, int64_t zp) {
  Tensor result = input;
  for (double& value : result.values)
    value = double(quantize(float(value), s, zp));
  return result;
}
inline Tensor output_tensor(const Json& spec, size_t index) {
  const std::string name = spec.at("args").at(index).at("tensor");
  auto sizes =
      spec.at("tensors").at(name).at("sizes").get<std::vector<int64_t>>();
  size_t count = 1;
  for (int64_t size : sizes)
    count *= size_t(size);
  return {std::move(sizes), std::vector<double>(count)};
}
inline size_t broadcast_index(
    size_t index,
    const std::vector<int64_t>& out,
    const std::vector<int64_t>& sizes) {
  size_t source = 0, stride = 1;
  for (size_t i = 0; i < out.size(); ++i) {
    const size_t dim = out.size() - 1 - i;
    const size_t coordinate = index % size_t(out[dim]);
    index /= size_t(out[dim]);
    if (i < sizes.size()) {
      const size_t source_dim = sizes.size() - 1 - i;
      if (sizes[source_dim] != 1)
        source += coordinate * stride;
      stride *= size_t(sizes[source_dim]);
    }
  }
  return source;
}
} // namespace vkcompute::shader_profiling::reference::q8ta
