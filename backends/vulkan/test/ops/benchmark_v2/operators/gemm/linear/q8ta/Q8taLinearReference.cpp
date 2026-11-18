// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>
#include <array>
namespace vkcompute::shader_profiling::reference {
static Tensor q8ta_linear_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Tensor& input = tensor_arg(spec, 0, tensors);
  const Tensor& weight = tensor_arg(spec, 3, tensors);
  const Tensor& weight_sums = tensor_arg(spec, 4, tensors);
  const Tensor& scales = tensor_arg(spec, 5, tensors);
  const Tensor* bias =
      spec.at("args").at(8).is_null() ? nullptr : &tensor_arg(spec, 8, tensors);
  require(
      !input.sizes.empty() && weight.sizes.size() == 2,
      "Q8 linear requires input rank >= 1 and matrix weights");
  const int64_t k = input.sizes.back(), n = weight.sizes[0];
  require(k > 0 && n > 0, "Q8 linear requires positive feature dimensions");
  const int64_t m = int64_t(input.values.size()) / k;
  require(weight.sizes[1] == k, "Q8 linear weight/input features disagree");
  auto out_sizes = input.sizes;
  out_sizes.back() = n;
  Tensor out{std::move(out_sizes), std::vector<double>(size_t(m * n))};
  const float is = q8ta::scale(spec, 1), os = q8ta::scale(spec, 6);
  const int64_t iz = q8ta::integer(spec, 2), oz = q8ta::integer(spec, 7);
  std::vector<int16_t> quantized(input.values.size());
  for (size_t i = 0; i < quantized.size(); ++i)
    quantized[i] = int16_t(input.values[i]);
  std::vector<int8_t> packed_weight(weight.values.size());
  for (int64_t col = 0; col < n; ++col)
    for (int64_t j = 0; j < k; ++j)
      packed_weight[size_t(j * n + col)] =
          int8_t(weight.values[size_t(col * k + j)]);
  constexpr int64_t tile = 32;
  const int64_t blocks = (n + tile - 1) / tile;
  parallel_for(size_t(m * blocks), threads, [&](size_t index) {
    const int64_t row = int64_t(index) / blocks,
                  start = int64_t(index) % blocks * tile,
                  count = std::min(tile, n - start);
    std::array<int64_t, tile> sums{};
    for (int64_t j = 0; j < k; ++j) {
      const int64_t value = quantized[size_t(row * k + j)];
      for (int64_t col = 0; col < count; ++col)
        sums[size_t(col)] += value * packed_weight[size_t(j * n + start + col)];
    }
    for (int64_t col = 0; col < count; ++col) {
      const size_t oc = size_t(start + col);
      const int64_t corrected =
          sums[size_t(col)] - iz * int64_t(weight_sums.values[oc]);
      out.values[size_t(row * n) + oc] = double(q8ta::requantize_accumulator(
          corrected,
          is,
          float(scales.values[oc]),
          os,
          oz,
          bias ? float(bias->values[oc]) : 0.0f,
          spec.at("args").at(9).at("string") == "relu",
          false));
    }
  });
  return out;
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q8ta_linear, q8ta_linear_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.q8ta_linear.default, q8ta_linear);
  REGISTER_OPERATOR_REFERENCE(et_vk.q8ta_linear_gemv.default, q8ta_linear);
}
} // namespace vkcompute::shader_profiling::reference
