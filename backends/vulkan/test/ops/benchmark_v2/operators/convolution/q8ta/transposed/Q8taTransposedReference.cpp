// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dReferenceCommon.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/q8ta/Q8taConv2dReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>

namespace vkcompute::shader_profiling::reference {
Tensor q8ta_transposed_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Tensor& input = tensor_arg(spec, 0, tensors);
  const Tensor& weight = tensor_arg(spec, 3, tensors);
  const Tensor& sums = tensor_arg(spec, 4, tensors);
  const Tensor& scales = tensor_arg(spec, 5, tensors);
  const Tensor* bias =
      spec.at("args").at(8).is_null() ? nullptr : &tensor_arg(spec, 8, tensors);
  Tensor out = q8ta::output_tensor(spec, 16);
  const auto kernel = conv2d_pair_arg(spec, 9);
  const auto stride = conv2d_pair_arg(spec, 10);
  const auto pad = conv2d_pair_arg(spec, 11);
  const auto output_padding = conv2d_pair_arg(spec, 12);
  const auto dilation = conv2d_pair_arg(spec, 13);
  const int64_t groups = q8ta::integer(spec, 14);
  require(
      input.sizes.size() == 4 && out.sizes.size() == 4 && groups > 0,
      "Q8 transposed convolution requires NCHW tensors and positive groups");
  const int64_t cin = input.sizes[1], cout = out.sizes[1];
  const int64_t height = out.sizes[2], width = out.sizes[3];
  require(
      cin % groups == 0 && cout % groups == 0,
      "Q8 transposed convolution channels must be divisible by groups");
  const int64_t columns = cin / groups * kernel[0] * kernel[1];
  require(
      weight.sizes.size() == 2 && weight.sizes[0] == cout &&
          weight.sizes[1] >= columns && sums.values.size() >= size_t(cout) &&
          scales.values.size() >= size_t(cout) &&
          (!bias || bias->values.size() >= size_t(cout)),
      "Q8 transposed convolution requires OHWI weights and channel parameters");
  for (size_t d = 0; d < 2; ++d) {
    require(
        kernel[d] > 0 && stride[d] > 0 && dilation[d] > 0 && pad[d] >= 0 &&
            output_padding[d] >= 0 &&
            (output_padding[d] < stride[d] || output_padding[d] < dilation[d]),
        "Q8 transposed convolution spatial parameters are invalid");
    require(
        out.sizes[d + 2] ==
            (input.sizes[d + 2] - 1) * stride[d] - 2 * pad[d] +
                dilation[d] * (kernel[d] - 1) + output_padding[d] + 1,
        "Q8 transposed convolution output shape disagrees with parameters");
  }
  require(out.sizes[0] == input.sizes[0], "Q8 transposed batch sizes disagree");
  const float is = q8ta::scale(spec, 1), os = q8ta::scale(spec, 6);
  const int64_t iz = q8ta::integer(spec, 2), oz = q8ta::integer(spec, 7);
  require(
      std::isfinite(is) && is > 0 && std::isfinite(os) && os > 0 &&
          iz >= -128 && iz <= 127 && oz >= -128 && oz <= 127,
      "Q8 transposed convolution quantization parameters are invalid");
  const std::string activation = spec.at("args").at(15).at("string");
  require(
      activation == "none" || activation == "relu",
      "Q8 transposed convolution reference supports none/relu");
  std::vector<int64_t> logical_sums(size_t(cout), 0);
  for (int64_t oc = 0; oc < cout; ++oc)
    for (int64_t k = 0; k < columns; ++k)
      logical_sums[size_t(oc)] +=
          int64_t(weight.values[size_t(oc * weight.sizes[1] + k)]);
  parallel_for(out.values.size(), threads, [&](size_t index) {
    const int64_t x = int64_t(index) % width;
    const int64_t y = int64_t(index) / width % height;
    const int64_t oc = int64_t(index) / (width * height) % cout;
    const int64_t n = int64_t(index) / (width * height * cout);
    const int64_t group = oc / (cout / groups);
    int64_t sum = 0;
    for (int64_t c = 0; c < cin / groups; ++c)
      for (int64_t ky = 0; ky < kernel[0]; ++ky)
        for (int64_t kx = 0; kx < kernel[1]; ++kx) {
          const int64_t iy = y + pad[0] - ky * dilation[0];
          const int64_t ix = x + pad[1] - kx * dilation[1];
          if (iy < 0 || ix < 0 || iy % stride[0] || ix % stride[1] ||
              iy / stride[0] >= input.sizes[2] ||
              ix / stride[1] >= input.sizes[3])
            continue;
          const int64_t source =
              ((n * cin + group * (cin / groups) + c) * input.sizes[2] +
               iy / stride[0]) *
                  input.sizes[3] +
              ix / stride[1];
          const int64_t wi =
              oc * weight.sizes[1] + (ky * kernel[1] + kx) * (cin / groups) + c;
          sum += (int64_t(input.values[size_t(source)]) - iz) *
              int64_t(weight.values[size_t(wi)]);
        }
    const int64_t raw = sum + iz * logical_sums[size_t(oc)];
    const float corrected =
        float(raw) - float(sums.values[size_t(oc)]) * float(iz);
    float value = corrected * (float(scales.values[size_t(oc)]) * is);
    value += bias ? float(bias->values[size_t(oc)]) : 0;
    if (activation == "relu")
      value = std::max(value, 0.0f);
    out.values[index] = double(q8ta::requantize(value, 1.0f / os, oz));
  });
  return out;
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(q8ta_transposed, q8ta_transposed_reference);
  REGISTER_OPERATOR_REFERENCE(
      et_vk.q8ta_conv2d_transposed.default, q8ta_transposed);
}
} // namespace vkcompute::shader_profiling::reference
