// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dRouteCommon.h>

#include <limits>

namespace vkcompute::shader_profiling::convolution {

void supported(bool condition, const std::string& message) {
  if (!condition) {
    throw UnsupportedRoute(message);
  }
}

void check_options(const Json& options, bool intermediate_storage) {
  if (!options.is_object()) {
    throw std::invalid_argument("Route options must be an object");
  }
  for (const auto& item : options.items()) {
    if (!intermediate_storage || item.key() != "intermediate_storage") {
      throw std::invalid_argument(
          "Unknown convolution route option: " + item.key());
    }
  }
}

std::array<int64_t, 2>
spatial_parameter(ComputeGraph& graph, ValueRef value, bool positive) {
  const std::vector<int64_t> values = *graph.get_int_list(value);
  supported(values.size() == 2, "Conv2d requires two spatial parameter values");
  for (int64_t item : values) {
    supported(
        item >= (positive ? 1 : 0) && item <= INT32_MAX,
        "Conv2d spatial parameters are out of range");
  }
  return {values[0], values[1]};
}

void constant_tensor(
    ComputeGraph& graph,
    ValueRef value,
    const std::string& name) {
  supported(graph.val_is_tref(value), name + " requires storage: constant");
}

int64_t product(int64_t a, int64_t b) {
  supported(
      a >= 0 && b >= 0 &&
          (b == 0 || a <= std::numeric_limits<int64_t>::max() / b),
      "Convolution route dimensions overflow int64");
  return a * b;
}

int64_t padded4(int64_t value) {
  supported(
      value > 0 && value <= INT32_MAX,
      "Convolution channel dimension is out of range");
  return (value + 3) / 4 * 4;
}

void grouped_packing(const ConvShape& shape) {
  if (shape.groups > 1) {
    supported(
        shape.input[1] / shape.groups % 4 == 0 &&
            shape.output[1] / shape.groups % 4 == 0,
        "Grouped convolution requires input/output channels per group divisible by 4");
  }
}

Json selected(const std::string& name, const Json& options) {
  return {{"name", name}, {"options", options}};
}

ConvShape conv_shape_base(
    ComputeGraph& graph,
    ValueRef input,
    ValueRef weight,
    ValueRef output,
    int64_t groups,
    size_t weight_rank) {
  supported(
      graph.val_is_tensor(input) && graph.val_is_tensor(output),
      "Convolution input/output must be GPU tensors");
  constant_tensor(graph, weight, "Convolution weights");
  ConvShape shape{
      graph.sizes_of(input),
      graph.sizes_of(output),
      graph.sizes_of(weight),
      {},
      groups};
  supported(
      shape.input.size() == 4 && shape.output.size() == 4 &&
          shape.weight.size() == weight_rank,
      "Convolution route supports rank-4 NCHW input/output and canonical weights");
  for (const auto* sizes : {&shape.input, &shape.output, &shape.weight}) {
    for (int64_t size : *sizes) {
      supported(
          size > 0 && size <= INT32_MAX,
          "Convolution tensor dimensions are out of range");
    }
  }
  supported(
      shape.groups > 0 && shape.groups <= INT32_MAX &&
          shape.input[1] % shape.groups == 0 &&
          shape.output[1] % shape.groups == 0,
      "Convolution channels must be divisible by positive groups");
  supported(
      shape.input[0] == shape.output[0] && shape.weight[0] == shape.output[1],
      "Convolution input/output/weight dimensions disagree");
  return shape;
}

void check_conv_output(
    ComputeGraph& graph,
    const ConvShape& shape,
    ValueRef stride_value,
    ValueRef padding_value,
    ValueRef dilation_value) {
  const auto stride = spatial_parameter(graph, stride_value, true);
  const auto padding = spatial_parameter(graph, padding_value, false);
  const auto dilation = spatial_parameter(graph, dilation_value, true);
  for (size_t i = 0; i < 2; ++i) {
    const int64_t numerator = shape.input[i + 2] + 2 * padding[i] -
        product(dilation[i], shape.kernel[i] - 1) - 1;
    supported(
        numerator >= 0 && numerator / stride[i] + 1 == shape.output[i + 2],
        "Convolution declared output shape disagrees with spatial parameters");
  }
}

} // namespace vkcompute::shader_profiling::convolution
