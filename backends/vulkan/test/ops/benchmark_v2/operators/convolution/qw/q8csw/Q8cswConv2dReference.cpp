// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dReferenceCommon.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>
#include <cmath>
namespace vkcompute::shader_profiling::reference {
static Tensor q8csw_conv2d_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const bool quantized =
      spec.at("operator") == "et_vk.conv2d_q8ta_q8csw.default";
  const size_t wi = quantized ? 3 : 1, si = quantized ? 5 : 2,
               bi = quantized ? 6 : 3, ki = quantized ? 7 : 4;
  Tensor input = tensor_arg(spec, 0, tensors);
  const Tensor& weight = tensor_arg(spec, wi, tensors);
  const Tensor& scales = tensor_arg(spec, si, tensors);
  const Tensor* bias = spec.at("args").at(bi).is_null()
      ? nullptr
      : &tensor_arg(spec, bi, tensors);
  const Conv2dParameters p{
      conv2d_pair_arg(spec, ki),
      conv2d_pair_arg(spec, ki + 1),
      conv2d_pair_arg(spec, ki + 2),
      conv2d_pair_arg(spec, ki + 3),
      q8ta::integer(spec, ki + 4)};
  const auto g = conv2d_geometry(input, weight, bias, p);
  const auto weight_index =
      [&](int64_t oc, int64_t ci, int64_t ky, int64_t kx) {
        return oc * weight.sizes[1] +
            (ky * p.kernel[1] + kx) * g.in_channels_per_group + ci;
      };
  if (quantized) {
    const float scale = q8ta::scale(spec, 1);
    const int64_t zp = q8ta::integer(spec, 2);
    input = q8ta::quantized_input(input, scale, zp);
    return convolve<int64_t>(
        input,
        weight,
        p,
        g,
        threads,
        weight_index,
        [&](int64_t sum, double value, double w) {
          return sum + (int64_t(value) - zp) * int64_t(w);
        },
        [&](int64_t sum, int64_t oc) {
          const float combined_scale = scale * float(scales.values[size_t(oc)]);
          return std::fma(float(sum), combined_scale, 0.0f) +
              (bias ? float(bias->values[size_t(oc)]) : 0.0f);
        });
  }
  return convolve<float>(
      input,
      weight,
      p,
      g,
      threads,
      weight_index,
      [](float sum, double value, double w) {
        return sum + float(value) * float(w);
      },
      [&](float sum, int64_t oc) {
        return sum * float(scales.values[size_t(oc)]) +
            (bias ? float(bias->values[size_t(oc)]) : 0.0f);
      });
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q8csw_conv2d, q8csw_conv2d_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.conv2d_q8csw.default, q8csw_conv2d);
  REGISTER_OPERATOR_REFERENCE(et_vk.conv2d_q8ta_q8csw.default, q8csw_conv2d);
}
} // namespace vkcompute::shader_profiling::reference
