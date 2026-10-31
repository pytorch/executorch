// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/GemmReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/linear/qw/dq8ca_q4gsw/Dq8caQ4gswReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/linear/qw/q4gsw/Q4gswLinearReference.h>
#include <algorithm>
#include <cmath>
namespace vkcompute::shader_profiling::reference {
Tensor dq8ca_q4gsw_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  Json canonical = spec;
  canonical["args"] = {
      spec.at("args").at(0),
      spec.at("args").at(3),
      spec.at("args").at(5),
      spec.at("args").at(6),
      spec.at("args").at(7),
      spec.at("args").at(8)};
  const Json context = spec.value("reference_context", Json::object());
  require(context.is_object(), "Reference context must be an object");
  const Json dot = context.value("int8_dot_product", Json(true));
  require(dot.is_boolean(), "DQ8CA reference capability must be boolean");
  if (!dot.get<bool>()) {
    return q4gsw_linear_reference(canonical, tensors, threads);
  }
  TensorMap converted = tensors;
  const std::string name = spec.at("args").at(0).at("tensor");
  Tensor& input = converted.at(name);
  const Tensor& scales = tensor_arg(spec, 1, tensors);
  const Tensor& zeros = tensor_arg(spec, 2, tensors);
  const size_t k = input.sizes.back(), m = input.values.size() / k;
  require(
      scales.values.size() == m && zeros.values.size() == m,
      "Dynamic Q4 activation qparams must be per row");
  parallel_for(m, threads, [&](size_t row) {
    const float scale = scales.values[row], zp = zeros.values[row];
    require(scale > 0, "Activation scale must be positive");
    for (size_t i = 0; i < k; ++i) {
      const float q = std::clamp(
          std::round(float(input.values[row * k + i]) / scale) + zp,
          -128.0f,
          127.0f);
      input.values[row * k + i] = (q - zp) * scale;
    }
  });
  return q4gsw_linear_reference(canonical, converted, threads);
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(dq8ca_q4gsw, dq8ca_q4gsw_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.linear_dq8ca_q4gsw.default, dq8ca_q4gsw);
}
} // namespace vkcompute::shader_profiling::reference
