// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/GemmReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/matmul/MatmulReference.h>
namespace vkcompute::shader_profiling::reference {
Tensor
matmul_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  const bool add = spec.at("operator") == "aten.addmm.default";
  const Tensor& a = tensor_arg(spec, add ? 1 : 0, tensors);
  const Tensor& b = tensor_arg(spec, add ? 2 : 1, tensors);
  require(
      (a.sizes.size() == 2 || a.sizes.size() == 3) &&
          a.sizes.size() == b.sizes.size(),
      "Matmul expects two rank2 or rank3 inputs");
  const size_t k = a.sizes.back(), m = a.sizes[a.sizes.size() - 2],
               n = b.sizes.back();
  const size_t batches = a.sizes.size() == 3 ? a.sizes[0] : 1;
  require(
      b.sizes[b.sizes.size() - 2] == int64_t(k) &&
          (a.sizes.size() == 2 || a.sizes[0] == b.sizes[0]),
      "Matmul dimensions differ");
  Tensor out{a.sizes, std::vector<double>(batches * m * n)};
  out.sizes.back() = n;
  const Tensor* bias = add ? &tensor_arg(spec, 0, tensors) : nullptr;
  const float beta =
      add ? spec.at("args").at(3).at("float").get<float>() : 0.0f;
  const float alpha =
      add ? spec.at("args").at(4).at("float").get<float>() : 1.0f;
  require(
      !bias || bias->values.size() == n,
      "Addmm reference requires vector bias");
  for (size_t batch = 0; batch < batches; ++batch) {
    Tensor input{
        {int64_t(m), int64_t(k)},
        std::vector<double>(
            a.values.begin() + batch * m * k,
            a.values.begin() + (batch + 1) * m * k)};
    Tensor weight{{int64_t(n), int64_t(k)}, std::vector<double>(n * k)};
    parallel_for(n, threads, [&](size_t j) {
      for (size_t i = 0; i < k; ++i)
        weight.values[j * k + i] = b.values[(batch * k + i) * n + j];
    });
    Tensor result = linear_projection(input, weight, nullptr, threads);
    for (size_t i = 0; i < m * n; ++i)
      out.values[batch * m * n + i] = alpha * float(result.values[i]) +
          (bias ? beta * float(bias->values[i % n]) : 0.0f);
  }
  return out;
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(matmul, matmul_reference);
  REGISTER_OPERATOR_REFERENCE(aten.mm.default, matmul);
  REGISTER_OPERATOR_REFERENCE(aten.bmm.default, matmul);
  REGISTER_OPERATOR_REFERENCE(aten.addmm.default, matmul);
}
} // namespace vkcompute::shader_profiling::reference
