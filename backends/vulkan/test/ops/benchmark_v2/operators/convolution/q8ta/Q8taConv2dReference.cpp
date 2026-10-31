// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dReferenceCommon.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/q8ta/Q8taConv2dReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>

#include <algorithm>
#include <cmath>

namespace vkcompute::shader_profiling::reference {
namespace {
bool fused_bias(const Json& spec) {
  const Json context = spec.value("reference_context", Json::object());
  if (context.contains("shaders")) {
    for (const Json& node : context.at("shaders")) {
      if (node.get<std::string>().rfind("q8ta_conv2d_pw", 0) == 0)
        return true;
    }
    return false;
  }
  const Json route = spec.value("route", Json("auto"));
  const std::string name = route.is_string() ? route.get<std::string>()
                                             : route.value("name", "auto");
  return spec.at("operator") == "et_vk.q8ta_conv2d_pw.default" ||
      name.rfind("im2col", 0) == 0;
}
} // namespace

Tensor q8ta_conv2d_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Conv2dParameters p{
      conv2d_pair_arg(spec, 9),
      conv2d_pair_arg(spec, 10),
      conv2d_pair_arg(spec, 11),
      conv2d_pair_arg(spec, 12),
      q8ta::integer(spec, 13)};
  const float input_scale = q8ta::scale(spec, 1);
  const float output_scale = q8ta::scale(spec, 6);
  const int64_t input_zp = q8ta::integer(spec, 2);
  const int64_t output_zp = q8ta::integer(spec, 7);
  require(
      std::isfinite(input_scale) && input_scale > 0 &&
          std::isfinite(output_scale) && output_scale > 0,
      "Q8 convolution scales must be positive and finite");
  require(
      input_zp >= -128 && input_zp <= 127 && output_zp >= -128 &&
          output_zp <= 127,
      "Q8 convolution zero points must fit int8");
  const bool legacy = spec.at("args").size() == 15;
  const std::string activation =
      legacy ? "none" : spec.at("args").at(14).at("string").get<std::string>();
  require(
      activation == "none" || activation == "relu",
      "Q8 convolution reference supports none/relu activation");
  const Tensor& input = tensor_arg(spec, 0, tensors);
  const Tensor& weight = tensor_arg(spec, 3, tensors);
  const Tensor& sums = tensor_arg(spec, 4, tensors);
  const Tensor& scales = tensor_arg(spec, 5, tensors);
  const Tensor* bias =
      spec.at("args").at(8).is_null() ? nullptr : &tensor_arg(spec, 8, tensors);
  const std::string output_name = spec.at("args").back().at("tensor");
  const auto output_sizes = spec.at("tensors")
                                .at(output_name)
                                .at("sizes")
                                .get<std::vector<int64_t>>();
  require(output_sizes.size() == 4, "Q8 convolution output must be NCHW");
  const Tensor geometry_weight{{output_sizes[1]}, {}};
  const auto g = conv2d_geometry(input, geometry_weight, nullptr, p);
  require(
      output_sizes ==
          std::vector<int64_t>{
              g.batches, g.out_channels, g.out_height, g.out_width},
      "Q8 convolution output shape disagrees with parameters");
  const bool depthwise = weight.sizes.size() == 3;
  require(
      depthwise
          ? (weight.sizes ==
                 std::vector<int64_t>{
                     p.kernel[0], p.kernel[1], g.out_channels} &&
             g.channels == g.out_channels && p.groups == g.channels)
          : (weight.sizes.size() == 2 && weight.sizes[0] == g.out_channels &&
             weight.sizes[1] >= g.row_size),
      "Q8 convolution weights require OHWI rows or depthwise HWC");
  require(
      sums.values.size() >= size_t(g.out_channels) &&
          scales.values.size() >= size_t(g.out_channels) &&
          (!bias || bias->values.size() >= size_t(g.out_channels)),
      "Q8 convolution requires per-output-channel sums/scales/bias");
  const auto weight_index =
      [&](int64_t oc, int64_t ci, int64_t ky, int64_t kx) {
        return depthwise ? (ky * p.kernel[1] + kx) * g.out_channels + oc
                         : oc * weight.sizes[1] +
                (ky * p.kernel[1] + kx) * g.in_channels_per_group + ci;
      };
  std::vector<int64_t> logical_sums(size_t(g.out_channels), 0);
  for (int64_t oc = 0; oc < g.out_channels; ++oc)
    for (int64_t ci = 0; ci < g.in_channels_per_group; ++ci)
      for (int64_t ky = 0; ky < p.kernel[0]; ++ky)
        for (int64_t kx = 0; kx < p.kernel[1]; ++kx)
          logical_sums[size_t(oc)] +=
              int64_t(weight.values[size_t(weight_index(oc, ci, ky, kx))]);
  const bool fused = !legacy && fused_bias(spec);
  return convolve<int64_t>(
      input,
      weight,
      p,
      g,
      threads,
      weight_index,
      [&](int64_t sum, double value, double weight_value) {
        return sum + (int64_t(value) - input_zp) * int64_t(weight_value);
      },
      [&](int64_t sum, int64_t oc) {
        const int64_t raw = sum + input_zp * logical_sums[size_t(oc)];
        const float corrected = fused
            ? float(raw - input_zp * int64_t(sums.values[size_t(oc)]))
            : float(raw) - float(sums.values[size_t(oc)]) * float(input_zp);
        const float multiplier = float(scales.values[size_t(oc)]) * input_scale;
        const float channel_bias = bias ? float(bias->values[size_t(oc)]) : 0;
        const float scaled = corrected * multiplier;
        float value = fused ? std::fma(corrected, multiplier, channel_bias)
                            : scaled + channel_bias;
        if (activation == "relu")
          value = std::max(value, 0.0f);
        return double(q8ta::requantize(value, 1.0f / output_scale, output_zp));
      });
}

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q8ta_conv2d, q8ta_conv2d_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.q8ta_conv2d.default, q8ta_conv2d);
  REGISTER_OPERATOR_REFERENCE(et_vk.q8ta_conv2d_general.default, q8ta_conv2d);
  REGISTER_OPERATOR_REFERENCE(et_vk.q8ta_conv2d_im2col.default, q8ta_conv2d);
  REGISTER_OPERATOR_REFERENCE(et_vk.q8ta_conv2d_dw.default, q8ta_conv2d);
  REGISTER_OPERATOR_REFERENCE(et_vk.q8ta_conv2d_pw.default, q8ta_conv2d);
  REGISTER_OPERATOR_REFERENCE(
      et_vk.conv2d_q8ta_q8csw_q8to.default, q8ta_conv2d);
}
} // namespace vkcompute::shader_profiling::reference
