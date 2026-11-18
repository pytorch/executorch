// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>

namespace vkcompute::shader_profiling::reference {
static Tensor q8ta_pixel_shuffle_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Tensor& input = tensor_arg(spec, 0, tensors);
  Tensor out = q8ta::output_tensor(spec, spec.at("args").size() - 1);
  const float is = q8ta::scale(spec, 1),
              inverse_out_scale = q8ta::scale(spec, 3);
  const int64_t iz = q8ta::integer(spec, 2), oz = q8ta::integer(spec, 4),
                r = q8ta::integer(spec, 5);
  const float actual_out_scale = 1.0f / inverse_out_scale;
  const bool passthrough = std::abs(is - actual_out_scale) <=
          1e-7f * std::max(std::abs(is), 1e-7f) &&
      iz == oz;
  const int64_t channels = out.sizes[1], height = out.sizes[2],
                width = out.sizes[3];
  parallel_for(out.values.size(), threads, [&](size_t i) {
    const int64_t x = int64_t(i) % width, y = int64_t(i) / width % height;
    const int64_t c = int64_t(i) / (width * height) % channels,
                  n = int64_t(i) / (width * height * channels);
    const int64_t source_c = c * r * r + y % r * r + x % r;
    const int64_t source =
        ((n * input.sizes[1] + source_c) * input.sizes[2] + y / r) *
            input.sizes[3] +
        x / r;
    const int64_t value = int64_t(input.values[source]);
    out.values[i] = passthrough
        ? double(value)
        : double(q8ta::requantize(
              q8ta::dequantize(value, is, iz), inverse_out_scale, oz));
  });
  return out;
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q8ta_pixel_shuffle, q8ta_pixel_shuffle_reference);
  REGISTER_OPERATOR_REFERENCE(
      et_vk.q8ta_pixel_shuffle.default, q8ta_pixel_shuffle);
}
} // namespace vkcompute::shader_profiling::reference
