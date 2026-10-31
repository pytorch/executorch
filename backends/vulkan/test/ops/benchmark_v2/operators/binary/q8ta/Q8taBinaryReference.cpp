// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>

namespace vkcompute::shader_profiling::reference {
static Tensor
q8ta_add_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  const Tensor& a = tensor_arg(spec, 0, tensors);
  const Tensor& b = tensor_arg(spec, 1, tensors);
  Tensor out = q8ta::output_tensor(spec, spec.at("args").size() - 1);
  const float as = q8ta::scale(spec, 2), bs = q8ta::scale(spec, 4),
              os = q8ta::scale(spec, 6), alpha = q8ta::scale(spec, 8);
  const int64_t az = q8ta::integer(spec, 3), bz = q8ta::integer(spec, 5),
                oz = q8ta::integer(spec, 7);
  parallel_for(out.values.size(), threads, [&](size_t i) {
    const int64_t qa =
        int64_t(a.values[q8ta::broadcast_index(i, out.sizes, a.sizes)]);
    const int64_t qb =
        int64_t(b.values[q8ta::broadcast_index(i, out.sizes, b.sizes)]);
    const float value =
        q8ta::dequantize(qa, as, az) + alpha * q8ta::dequantize(qb, bs, bz);
    out.values[i] = double(q8ta::quantize(value, os, oz));
  });
  return out;
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q8ta_add, q8ta_add_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.q8ta_add.default, q8ta_add);
}
} // namespace vkcompute::shader_profiling::reference
