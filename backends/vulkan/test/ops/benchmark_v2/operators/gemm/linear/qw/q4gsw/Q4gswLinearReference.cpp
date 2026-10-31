// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/GemmReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/linear/qw/q4gsw/Q4gswLinearReference.h>

namespace vkcompute::shader_profiling::reference {

Tensor q4gsw_linear_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Tensor& input = tensor_arg(spec, 0, tensors);
  const Tensor& packed = tensor_arg(spec, 1, tensors);
  const Tensor& scales = tensor_arg(spec, 2, tensors);
  require(
      !input.sizes.empty() && packed.sizes.size() == 2,
      "Q4 linear requires input rank >= 1 and packed weight rank 2");
  const size_t k = input.sizes.back();
  const size_t n = packed.sizes[0];
  const int64_t group_size = spec.at("args").at(3).at("int");
  require(
      group_size > 0 && k % group_size == 0 && k % 2 == 0,
      "Q4 linear requires even K divisible by group_size");
  require(
      size_t(packed.sizes[1]) == k / 2,
      "Q4 packed weight shape must be [N, K/2]");
  require(
      scales.sizes ==
          std::vector<int64_t>({int64_t(k / group_size), int64_t(n)}),
      "Q4 scale shape must be [K/group_size, N]");
  Tensor weight{{int64_t(n), int64_t(k)}, std::vector<double>(n * k)};
  parallel_for(n, threads, [&](size_t column) {
    for (size_t inner = 0; inner < k; ++inner) {
      const uint8_t byte = packed.values[column * (k / 2) + inner / 2];
      const int value = ((byte >> (4 * (inner % 2))) & 15) - 8;
      weight.values[column * k + inner] = float(value) *
          float(scales.values[(inner / group_size) * n + column]);
    }
  });
  const Tensor* bias =
      spec.at("args").at(4).is_null() ? nullptr : &tensor_arg(spec, 4, tensors);
  return linear_projection(input, weight, bias, threads);
}

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q4gsw_linear, q4gsw_linear_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.q4gsw_linear.default, q4gsw_linear);
  REGISTER_OPERATOR_REFERENCE(et_vk.linear_q4gsw.default, q4gsw_linear);
}

} // namespace vkcompute::shader_profiling::reference
