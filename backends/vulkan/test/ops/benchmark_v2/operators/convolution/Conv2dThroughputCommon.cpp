// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dThroughputCommon.h>

#include <cmath>
#include <stdexcept>

namespace vkcompute::shader_profiling::throughput {

double convolution_flops(
    const Json& spec,
    size_t weight_index,
    size_t bias_index,
    const std::vector<int64_t>& weight_sizes,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups) {
  const Json& input = tensor_argument(spec, 0);
  const Json& weight = tensor_argument(spec, weight_index);
  const Json& output = output_tensor(spec);
  const auto input_sizes = input.at("sizes").get<std::vector<int64_t>>();
  const auto output_sizes = output.at("sizes").get<std::vector<int64_t>>();
  tensor_elements(input);
  tensor_elements(weight);
  const double output_elements = tensor_elements(output);
  if (input_sizes.size() != 4 || output_sizes.size() != 4 ||
      weight_sizes.size() != 4u) {
    throw std::invalid_argument(
        "Conv2d work count requires rank-4 input/output and supported weight shape");
  }
  const std::vector<int64_t> kernel{weight_sizes[2], weight_sizes[3]};
  if (groups <= 0 || input_sizes[1] % groups != 0 ||
      output_sizes[1] % groups != 0 || stride.size() != 2 ||
      padding.size() != 2 || dilation.size() != 2 ||
      input_sizes[0] != output_sizes[0] || weight_sizes[0] != output_sizes[1]) {
    throw std::invalid_argument(
        "Conv2d work count dimensions/groups/parameters disagree");
  }
  for (size_t i = 0; i < 2; ++i) {
    if (kernel[i] <= 0 || stride[i] <= 0 || padding[i] < 0 ||
        dilation[i] <= 0) {
      throw std::invalid_argument(
          "Conv2d work count requires positive spatial parameters");
    }
    const double expected =
        std::floor(
            (double(input_sizes[i + 2]) + 2 * double(padding[i]) -
             double(dilation[i]) * double(kernel[i] - 1) - 1) /
            double(stride[i])) +
        1;
    if (expected != double(output_sizes[i + 2])) {
      throw std::invalid_argument(
          "Conv2d work count output shape does not match stride/padding/dilation");
    }
  }
  const double inner =
      double(input_sizes[1] / groups) * double(kernel[0]) * double(kernel[1]);
  if (weight_sizes[1] != input_sizes[1] / groups) {
    throw std::invalid_argument(
        "Conv2d work count weight channels/kernel dimensions disagree");
  }
  const bool bias = !spec.at("args").at(bias_index).is_null();
  if (bias &&
      tensor_elements(tensor_argument(spec, bias_index)) != output_sizes[1]) {
    throw std::invalid_argument(
        "Conv2d work count bias must match output channels");
  }
  return output_elements * (2 * inner + (bias ? 1 : 0));
}

} // namespace vkcompute::shader_profiling::throughput
