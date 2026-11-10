// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dReferenceCommon.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/conv1d/Conv1dReference.h>

#include <algorithm>
#include <cmath>

namespace vkcompute::shader_profiling::reference {

namespace {

// The weight is [Cin, Cout / groups, Kh, Kw]. Each output pixel gathers the
// input pixels whose strided position lands on one of its kernel taps.
Tensor conv_transpose2d_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Tensor& input = tensor_arg(spec, 0, tensors);
  const Tensor& weight = tensor_arg(spec, 1, tensors);
  const Tensor* bias =
      spec.at("args").at(2).is_null() ? nullptr : &tensor_arg(spec, 2, tensors);
  const auto stride = conv2d_pair_arg(spec, 3);
  const auto padding = conv2d_pair_arg(spec, 4);
  const auto dilation = conv2d_pair_arg(spec, 5);
  const auto output_padding = conv2d_pair_arg(spec, 7);
  const int64_t groups = spec.at("args").at(8).at("int");
  require(
      input.sizes.size() == 4 && weight.sizes.size() == 4 &&
          weight.sizes[0] == input.sizes[1] && groups > 0 &&
          input.sizes[1] % groups == 0,
      "Transposed conv2d weight shape must be [Cin, Cout/groups, Kh, Kw]");
  const int64_t batches = input.sizes[0];
  const int64_t channels = input.sizes[1];
  const int64_t height = input.sizes[2];
  const int64_t width = input.sizes[3];
  const int64_t in_group = channels / groups;
  const int64_t out_group = weight.sizes[1];
  const int64_t out_channels = out_group * groups;
  const int64_t kh = weight.sizes[2];
  const int64_t kw = weight.sizes[3];
  require(
      !bias || bias->values.size() == size_t(out_channels),
      "Conv2d bias size must equal output channels");
  const int64_t out_height = (height - 1) * stride[0] - 2 * padding[0] +
      dilation[0] * (kh - 1) + output_padding[0] + 1;
  const int64_t out_width = (width - 1) * stride[1] - 2 * padding[1] +
      dilation[1] * (kw - 1) + output_padding[1] + 1;
  Tensor output{
      {batches, out_channels, out_height, out_width},
      std::vector<double>(
          size_t(batches * out_channels * out_height * out_width))};
  const size_t rows = size_t(batches * out_channels * out_height);
  parallel_for(rows, threads, [&](size_t row) {
    const int64_t oy = row % out_height;
    const int64_t oc = (row / out_height) % out_channels;
    const int64_t batch = row / (out_height * out_channels);
    const int64_t group = oc / out_group;
    for (int64_t ox = 0; ox < out_width; ++ox) {
      float sum = bias ? float(bias->values[oc]) : 0.0f;
      for (int64_t ci = 0; ci < in_group; ++ci) {
        const int64_t ic = group * in_group + ci;
        for (int64_t ky = 0; ky < kh; ++ky) {
          const int64_t y = oy + padding[0] - ky * dilation[0];
          if (y < 0 || y % stride[0] != 0 || y / stride[0] >= height)
            continue;
          for (int64_t kx = 0; kx < kw; ++kx) {
            const int64_t x = ox + padding[1] - kx * dilation[1];
            if (x < 0 || x % stride[1] != 0 || x / stride[1] >= width)
              continue;
            const double value =
                input.values
                    [((batch * channels + ic) * height + y / stride[0]) *
                         width +
                     x / stride[1]];
            const double weight_value =
                weight.values
                    [((ic * out_group + oc % out_group) * kh + ky) * kw + kx];
            sum = std::fma(float(value), float(weight_value), sum);
          }
        }
      }
      output.values[row * out_width + ox] = sum;
    }
  });
  return output;
}

} // namespace

Tensor
conv2d_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  if (tensor_arg(spec, 0, tensors).sizes.size() == 3) {
    return conv1d_reference(spec, tensors, threads);
  }
  if (spec.at("args").at(6).at("bool").get<bool>()) {
    return conv_transpose2d_reference(spec, tensors, threads);
  }
  const Tensor& input = tensor_arg(spec, 0, tensors);
  const Tensor& weight = tensor_arg(spec, 1, tensors);
  require(weight.sizes.size() == 4, "Conv2d reference requires rank-4 weight");
  const Conv2dParameters p{
      {weight.sizes[2], weight.sizes[3]},
      conv2d_pair_arg(spec, 3),
      conv2d_pair_arg(spec, 4),
      conv2d_pair_arg(spec, 5),
      spec.at("args").at(8).at("int")};
  const Tensor* bias =
      spec.at("args").at(2).is_null() ? nullptr : &tensor_arg(spec, 2, tensors);
  const auto g = conv2d_geometry(input, weight, bias, p);
  require(
      weight.sizes ==
          std::vector<int64_t>(
              {g.out_channels,
               g.in_channels_per_group,
               p.kernel[0],
               p.kernel[1]}),
      "Conv2d weight shape must be [Cout, Cin/groups, Kh, Kw]");
  return convolve<float>(
      input,
      weight,
      p,
      g,
      threads,
      [&](int64_t oc, int64_t ci, int64_t ky, int64_t kx) {
        return ((oc * g.in_channels_per_group + ci) * p.kernel[0] + ky) *
            p.kernel[1] +
            kx;
      },
      [](float sum, double value, double weight_value) {
        return std::fma(float(value), float(weight_value), sum);
      },
      [&](float sum, int64_t oc) {
        return sum + (bias ? float(bias->values[oc]) : 0.0f);
      });
}

namespace {

Tensor conv_with_clamp_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const float min = spec.at("args").at(9).at("float");
  const float max = spec.at("args").at(10).at("float");
  Tensor output = conv2d_reference(spec, tensors, threads);
  for (double& value : output.values) {
    value = std::clamp(float(value), min, max);
  }
  return output;
}

// Arguments after those of aten.convolution: residual, residual_scale,
// residual_op, alpha, activation, activation_a, activation_b, output.
Tensor conv2d_with_epilogue_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  const Json& args = spec.at("args");
  Tensor output = conv2d_reference(spec, tensors, threads);
  const int64_t channels = output.sizes.at(1);
  const int64_t plane = output.sizes.at(2) * output.sizes.at(3);
  require(
      args.at(10).is_null() || !args.at(9).is_null(),
      "Conv2d epilogue residual scale requires a residual");
  if (!args.at(9).is_null()) {
    const Tensor& residual = tensor_arg(spec, 9, tensors);
    require(
        residual.sizes == output.sizes,
        "Conv2d epilogue residual must match the output sizes");
    const Tensor* scale =
        args.at(10).is_null() ? nullptr : &tensor_arg(spec, 10, tensors);
    const size_t scale_rank = scale ? scale->sizes.size() : 0;
    require(
        !scale ||
            ((scale_rank == 3 || scale_rank == 4) &&
             scale->sizes[scale_rank - 3] == channels &&
             scale->values.size() == size_t(channels)),
        "Conv2d epilogue residual scale must be [1, C, 1, 1] or [C, 1, 1]");
    const std::string op = args.at(11).at("string");
    require(
        op == "add" || op == "sub" || op == "mul",
        "Conv2d epilogue residual op must be add, sub or mul");
    const float alpha = args.at(12).at("float");
    for (size_t i = 0; i < output.values.size(); ++i) {
      float r = float(residual.values[i]);
      if (scale) {
        r *= float(scale->values[(int64_t(i) / plane) % channels]);
      }
      const float v = float(output.values[i]);
      output.values[i] = op == "add" ? v + alpha * r
          : op == "sub"              ? v - alpha * r
                                     : v * r;
    }
  }
  const std::string activation = args.at(13).at("string");
  require(
      activation == "none" || activation == "relu" || activation == "clamp" ||
          activation == "leaky_relu",
      "Conv2d epilogue activation must be none, relu, clamp or leaky_relu");
  if (activation == "none") {
    return output;
  }
  const float a = args.at(14).at("float");
  const float b = args.at(15).at("float");
  for (double& value : output.values) {
    const float v = float(value);
    value = activation == "relu" ? std::max(v, 0.0f)
        : activation == "clamp"  ? std::clamp(v, a, b)
        : v < 0                  ? a * v
                                 : v;
  }
  return output;
}

} // namespace

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(conv2d, conv2d_reference);
  REGISTER_REFERENCE_FN(conv_with_clamp, conv_with_clamp_reference);
  REGISTER_OPERATOR_REFERENCE(aten.convolution.default, conv2d);
  REGISTER_REFERENCE_FN(conv2d_with_epilogue, conv2d_with_epilogue_reference);
  REGISTER_OPERATOR_REFERENCE(et_vk.conv_with_clamp.default, conv_with_clamp);
  REGISTER_OPERATOR_REFERENCE(et_vk.conv2d.with_epilogue, conv2d_with_epilogue);
}

} // namespace vkcompute::shader_profiling::reference
