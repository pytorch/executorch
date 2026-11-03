// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/GemmReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>
#include <array>
#include <cmath>
namespace vkcompute::shader_profiling::reference {
static Tensor q8csw_linear_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const bool quantized =
      spec.at("operator") == "et_vk.linear_q8ta_q8csw.default";
  const size_t wi = quantized ? 3 : 1, si = quantized ? 5 : 2,
               bi = quantized ? 6 : 3;
  Tensor input = tensor_arg(spec, 0, tensors),
         weight = tensor_arg(spec, wi, tensors);
  const Tensor& scales = tensor_arg(spec, si, tensors);
  const Tensor* bias = spec.at("args").at(bi).is_null()
      ? nullptr
      : &tensor_arg(spec, bi, tensors);
  if (quantized) {
    const float scale = q8ta::scale(spec, 1);
    const int64_t zp = q8ta::integer(spec, 2);
    require(
        weight.sizes.size() == 2 && input.sizes.back() == weight.sizes[1],
        "Quantized linear weight dimensions must match input features");
    const int64_t k = input.sizes.back(), n = weight.sizes[0];
    const int64_t m = int64_t(input.values.size()) / k;
    require(
        scales.values.size() == size_t(n) &&
            (!bias || bias->values.size() == size_t(n)),
        "Quantized linear scales and bias must match output features");
    Tensor output{input.sizes, std::vector<double>(size_t(m * n))};
    output.sizes.back() = n;
    std::vector<int16_t> quantized_input(input.values.size());
    for (size_t i = 0; i < quantized_input.size(); ++i)
      quantized_input[i] =
          int16_t(q8ta::quantize(float(input.values[i]), scale, zp) - zp);
    std::vector<int8_t> packed_weight(weight.values.size());
    for (int64_t col = 0; col < n; ++col)
      for (int64_t j = 0; j < k; ++j)
        packed_weight[size_t(j * n + col)] =
            int8_t(weight.values[size_t(col * k + j)]);
    constexpr int64_t tile = 32;
    const int64_t blocks = (n + tile - 1) / tile;
    parallel_for(size_t(m * blocks), threads, [&](size_t index) {
      const int64_t row = int64_t(index) / blocks;
      const int64_t start = int64_t(index) % blocks * tile;
      const int64_t count = std::min(tile, n - start);
      std::array<int64_t, tile> sums{};
      for (int64_t j = 0; j < k; ++j) {
        const int64_t value = quantized_input[size_t(row * k + j)];
        for (int64_t col = 0; col < count; ++col)
          sums[size_t(col)] +=
              value * packed_weight[size_t(j * n + start + col)];
      }
      for (int64_t col = 0; col < count; ++col) {
        const size_t oc = size_t(start + col);
        const float combined_scale = scale * float(scales.values[oc]);
        output.values[size_t(row * n) + oc] =
            std::fma(float(sums[size_t(col)]), combined_scale, 0.0f) +
            (bias ? float(bias->values[oc]) : 0.0f);
      }
    });
    return output;
  }
  for (size_t i = 0; i < weight.values.size(); ++i)
    weight.values[i] = float(weight.values[i]) *
        float(scales.values[i / size_t(weight.sizes[1])]);
  return linear_projection(input, weight, bias, threads);
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q8csw_linear, q8csw_linear_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.linear_q8csw.default, q8csw_linear);
  REGISTER_OPERATOR_REFERENCE(et_vk.linear_q8ta_q8csw.default, q8csw_linear);
}
} // namespace vkcompute::shader_profiling::reference
