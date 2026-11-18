// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>

namespace vkcompute::shader_profiling::reference {
static Tensor q8ta_relu_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  Tensor out = tensor_arg(spec, 0, tensors);
  const float in_scale = q8ta::scale(spec, 1), out_scale = q8ta::scale(spec, 3);
  const int64_t in_zp = q8ta::integer(spec, 2), out_zp = q8ta::integer(spec, 4);
  parallel_for(out.values.size(), threads, [&](size_t i) {
    const float value =
        q8ta::dequantize(int64_t(out.values[i]), in_scale, in_zp);
    out.values[i] =
        double(q8ta::quantize(std::max(value, 0.0f), out_scale, out_zp));
  });
  return out;
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q8ta_relu, q8ta_relu_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.q8ta_relu.default, q8ta_relu);
}
} // namespace vkcompute::shader_profiling::reference
