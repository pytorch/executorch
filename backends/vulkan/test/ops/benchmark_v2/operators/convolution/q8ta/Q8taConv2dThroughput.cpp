// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {
double q8ta_conv2d_flops(const Json& spec) {
  const auto input = throughput::tensor_argument(spec, 0)
                         .at("sizes")
                         .get<std::vector<int64_t>>();
  const auto weight = throughput::tensor_argument(spec, 3)
                          .at("sizes")
                          .get<std::vector<int64_t>>();
  const Json& output_tensor = throughput::output_tensor(spec);
  const auto output = output_tensor.at("sizes").get<std::vector<int64_t>>();
  const auto kernel =
      spec.at("args").at(9).at("int_list").get<std::vector<int64_t>>();
  const bool transposed =
      spec.value("operator_name", spec.value("operator", "")) ==
      "et_vk.q8ta_conv2d_transposed.default";
  const int64_t groups = spec.at("args").at(transposed ? 14 : 13).at("int");
  if (input.size() != 4 || output.size() != 4 || kernel.size() != 2 ||
      groups <= 0 || input[1] % groups || output[1] % groups ||
      input[0] != output[0] || kernel[0] <= 0 || kernel[1] <= 0)
    throw std::invalid_argument(
        "Q8 convolution work count dimensions disagree");
  const int64_t logical_columns = input[1] / groups * kernel[0] * kernel[1];
  const int64_t padded_columns = (logical_columns + 3) / 4 * 4;
  if (weight.size() == 3) {
    if (weight != std::vector<int64_t>{kernel[0], kernel[1], output[1]} ||
        groups != input[1] || input[1] != output[1])
      throw std::invalid_argument(
          "Q8 depthwise work count requires HWC weights");
  } else if (
      weight.size() != 2 || weight[0] != output[1] ||
      weight[1] != padded_columns) {
    throw std::invalid_argument("Q8 work count requires aligned OHWI rows");
  }
  const auto stride =
      spec.at("args").at(10).at("int_list").get<std::vector<int64_t>>();
  const auto pad =
      spec.at("args").at(11).at("int_list").get<std::vector<int64_t>>();
  const auto dilation = spec.at("args")
                            .at(transposed ? 13 : 12)
                            .at("int_list")
                            .get<std::vector<int64_t>>();
  const auto output_pad = transposed
      ? spec.at("args").at(12).at("int_list").get<std::vector<int64_t>>()
      : std::vector<int64_t>{0, 0};
  if (stride.size() != 2 || pad.size() != 2 || dilation.size() != 2 ||
      output_pad.size() != 2)
    throw std::invalid_argument("Q8 work count requires spatial pairs");
  for (size_t d = 0; d < 2; ++d) {
    if (stride[d] <= 0 || pad[d] < 0 || dilation[d] <= 0)
      throw std::invalid_argument(
          "Q8 work count spatial parameters are invalid");
    const int64_t expected = transposed
        ? (input[d + 2] - 1) * stride[d] - 2 * pad[d] +
            dilation[d] * (kernel[d] - 1) + output_pad[d] + 1
        : (input[d + 2] + 2 * pad[d] - dilation[d] * (kernel[d] - 1) - 1) /
                stride[d] +
            1;
    if (expected != output[d + 2])
      throw std::invalid_argument("Q8 work count output shape disagrees");
  }
  const double elements = throughput::tensor_elements(output_tensor);
  throughput::tensor_elements(throughput::tensor_argument(spec, 0));
  throughput::tensor_elements(throughput::tensor_argument(spec, 3));
  const bool bias = !spec.at("args").at(8).is_null();
  if (bias &&
      throughput::tensor_elements(throughput::tensor_argument(spec, 8)) <
          double(output[1]))
    throw std::invalid_argument(
        "Q8 work count bias must cover output channels");
  const double multiply_adds = transposed
      ? 2.0 * double(input[0]) * double(input[1]) * double(input[2]) *
          double(input[3]) * double(output[1] / groups) * double(kernel[0]) *
          double(kernel[1])
      : elements * 2.0 * double(logical_columns);
  return multiply_adds + (bias ? elements : 0);
}
} // namespace

REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_conv2d.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_conv2d_general.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_conv2d_im2col.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_conv2d_dw.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_conv2d_pw.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_conv2d_transposed.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.conv2d_q8ta_q8csw_q8to.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(et_vk.q8ta_conv2d.default, q8ta_conv2d_flops);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.q8ta_conv2d_general.default, q8ta_conv2d_flops);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.q8ta_conv2d_im2col.default, q8ta_conv2d_flops);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.q8ta_conv2d_dw.default, q8ta_conv2d_flops);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.q8ta_conv2d_pw.default, q8ta_conv2d_flops);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.q8ta_conv2d_transposed.default, q8ta_conv2d_flops);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.conv2d_q8ta_q8csw_q8to.default, q8ta_conv2d_flops);
  REGISTER_SHADER_MEMORY_TRAFFIC(q8ta_conv2d, throughput::tensor_traffic);
  REGISTER_SHADER_MEMORY_TRAFFIC(q8ta_im2col, throughput::tensor_traffic);
  REGISTER_SHADER_MEMORY_TRAFFIC(conv2d_dw_q8ta, throughput::tensor_traffic);
}
} // namespace vkcompute::shader_profiling
