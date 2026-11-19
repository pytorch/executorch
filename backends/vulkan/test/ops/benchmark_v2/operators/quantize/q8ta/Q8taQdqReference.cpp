// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>

namespace vkcompute::shader_profiling::reference {
static Tensor q8ta_quantize_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  Tensor out = tensor_arg(spec, 0, tensors);
  const float s = q8ta::scale(spec, 1);
  const int64_t zp = q8ta::integer(spec, 2);
  const int64_t minimum = q8ta::integer(spec, 3),
                maximum = q8ta::integer(spec, 4);
  require(
      -128 <= minimum && minimum <= maximum && maximum <= 127,
      "Quantize reference requires signed int8 quantization bounds");
  parallel_for(out.values.size(), threads, [&](size_t i) {
    out.values[i] = double(std::clamp(
        q8ta::quantize(float(out.values[i]), s, zp), minimum, maximum));
  });
  return out;
}
static Tensor q8ta_dequantize_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  Tensor out = tensor_arg(spec, 0, tensors);
  const float s = q8ta::scale(spec, 1);
  const int64_t zp = q8ta::integer(spec, 2);
  parallel_for(out.values.size(), threads, [&](size_t i) {
    out.values[i] = q8ta::dequantize(int64_t(out.values[i]), s, zp);
  });
  return out;
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q8ta_quantize, q8ta_quantize_reference);
  REGISTER_REFERENCE_FN(q8ta_dequantize, q8ta_dequantize_reference);
  REGISTER_OPERATOR_REFERENCE(
      quantized_decomposed.quantize_per_tensor.default, q8ta_quantize);
  REGISTER_OPERATOR_REFERENCE(
      quantized_decomposed.dequantize_per_tensor.default, q8ta_dequantize);
}
} // namespace vkcompute::shader_profiling::reference
