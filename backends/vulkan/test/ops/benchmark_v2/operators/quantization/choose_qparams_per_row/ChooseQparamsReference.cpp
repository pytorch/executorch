// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantization/choose_qparams_per_row/ChooseQparamsReference.h>
#include <algorithm>
#include <cmath>
#include <limits>
namespace vkcompute::shader_profiling::reference {
ReferenceOutputs choose_qparams_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Tensor& input = tensor_arg(spec, 0, tensors);
  require(input.sizes.size() == 2, "Choose qparams expects [rows,columns]");
  const int32_t qmin = spec.at("args").at(1).at("int"),
                qmax = spec.at("args").at(2).at("int");
  require(qmin < qmax, "Quantization bounds must be increasing");
  const size_t rows = input.sizes[0], columns = input.sizes[1];
  std::vector<double> scales(rows), zeros(rows);
  parallel_for(rows, threads, [&](size_t row) {
    float lo = 0, hi = 0;
    for (size_t j = 0; j < columns; ++j) {
      const float value = input.values[row * columns + j];
      lo = std::min(lo, value);
      hi = std::max(hi, value);
    }
    double scale = (double(hi) - lo) / (qmax - qmin);
    if (float(scale) == 0 || std::isinf(1.0f / float(scale)))
      scale = 0.1;
    constexpr float threshold = 6.1e-5f;
    if (scale < threshold) {
      const float original = scale;
      scale = threshold;
      if (lo == 0)
        hi = threshold * (qmax - qmin);
      else if (hi == 0)
        lo = -threshold * (qmax - qmin);
      else {
        lo *= threshold / original;
        hi *= threshold / original;
      }
    }
    const double from_lo = qmin - lo / scale, from_hi = qmax - hi / scale;
    const double error_lo = std::abs(qmin) - std::abs(lo / scale),
                 error_hi = std::abs(qmax) - std::abs(hi / scale);
    const double initial = error_lo < error_hi ? from_lo : from_hi;
    scales[row] = float(scale);
    zeros[row] = std::clamp(
        double(std::nearbyint(float(initial))), double(qmin), double(qmax));
  });
  return {
      {spec.at("args").at(3).at("tensor").get<std::string>(),
       std::move(scales)},
      {spec.at("args").at(4).at("tensor").get<std::string>(),
       std::move(zeros)}};
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_OUTPUTS_FN(
      choose_qparams_per_row, choose_qparams_reference);
  REGISTER_OPERATOR_REFERENCE(
      etvk.choose_qparams_per_row.default, choose_qparams_per_row);
}
} // namespace vkcompute::shader_profiling::reference
