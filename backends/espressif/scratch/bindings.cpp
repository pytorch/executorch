/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>

#include <esp_nn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

template <std::size_t N>
void check_bounds(
    const std::array<int32_t, N>& values,
    const char* name,
    int32_t minimum = 1,
    int32_t maximum = std::numeric_limits<uint16_t>::max()) {
  if (std::any_of(values.begin(), values.end(), [=](int32_t value) {
        return value < minimum || value > maximum;
      })) {
    throw pybind11::value_error(
        std::string(name) + " must be in [" + std::to_string(minimum) + ", " +
        std::to_string(maximum) + "]");
  }
}

int64_t checked_product(std::initializer_list<int64_t> factors) {
  int64_t product = 1;
  for (const auto factor : factors) {
    if (factor > std::numeric_limits<int32_t>::max() / product) {
      throw pybind11::value_error(
          "scratch query intermediates exceed the int32 range");
    }
    product *= factor;
  }
  return product;
}

void check_conv_arithmetic(
    const std::array<int32_t, 3>& input,
    const std::array<int32_t, 3>& filter,
    const std::array<int32_t, 3>& output,
    const std::array<int32_t, 2>& stride,
    const std::array<int32_t, 2>& padding,
    int32_t channel_multiplier = 1) {
  const int64_t aligned_channels = (int64_t{input[2]} + 15) & ~int64_t{15};
  const int64_t output_channels =
      std::max(int64_t{output[2]}, int64_t{input[2]} * channel_multiplier);
  std::array<int64_t, 2> extent;
  for (std::size_t axis = 0; axis < extent.size(); ++axis) {
    extent[axis] =
        std::max(int64_t{input[axis]}, int64_t{output[axis]} * stride[axis]) +
        2 * int64_t{padding[axis]} + filter[axis];
  }
  // Bound padded copies, aligned filters and coordinates before vendor int32
  // arithmetic. S3 stages 16 windows plus 24 windows of panel work and a 16 KiB
  // panel. Four int32 channel arrays and 256 bytes cover parameters and
  // alignment.
  const auto input_bound =
      checked_product({extent[0], extent[1], aligned_channels});
  const auto filter_bound = checked_product(
      {int64_t{filter[0]} * aligned_channels + 15, filter[1], output_channels});
  const auto output_bound = checked_product(
      {output[0], output[1], std::max(aligned_channels, output_channels)});
  const auto window_bound =
      checked_product({filter[0], filter[1], aligned_channels});
  const auto bound = 2 * input_bound + 2 * filter_bound + output_bound +
      40 * window_bound + 16 * std::max(aligned_channels, output_channels) +
      16 * 1024 + 256;
  if (bound > std::numeric_limits<int32_t>::max()) {
    throw pybind11::value_error(
        "scratch query intermediates exceed the int32 range");
  }
}

int conv2d_scratch_size(
    std::array<int32_t, 3> input_dims,
    std::array<int32_t, 3> filter_dims,
    std::array<int32_t, 3> output_dims,
    std::array<int32_t, 2> stride,
    std::array<int32_t, 2> padding) {
  check_bounds(input_dims, "input_dims");
  check_bounds(filter_dims, "filter_dims");
  check_bounds(output_dims, "output_dims");
  check_bounds(stride, "stride");
  check_bounds(padding, "padding", 0);
  check_conv_arithmetic(input_dims, filter_dims, output_dims, stride, padding);
  const data_dims_t input{input_dims[0], input_dims[1], input_dims[2], 0};
  const data_dims_t filter{filter_dims[0], filter_dims[1], filter_dims[2], 0};
  const data_dims_t output{output_dims[0], output_dims[1], output_dims[2], 0};
  conv_params_t params{};
  params.stride = {stride[0], stride[1]};
  params.padding = {padding[0], padding[1]};
  params.dilation = {1, 1};
  return esp_nn_get_conv_scratch_size(&input, &filter, &output, &params);
}

int depthwise_conv2d_scratch_size(
    std::array<int32_t, 3> input_dims,
    std::array<int32_t, 3> filter_dims,
    std::array<int32_t, 3> output_dims,
    std::array<int32_t, 2> stride,
    std::array<int32_t, 2> padding,
    int32_t channel_multiplier) {
  check_bounds(input_dims, "input_dims");
  check_bounds(filter_dims, "filter_dims");
  check_bounds(output_dims, "output_dims");
  check_bounds(stride, "stride");
  check_bounds(padding, "padding", 0);
  check_bounds(std::array{channel_multiplier}, "channel_multiplier");
  check_conv_arithmetic(
      input_dims,
      filter_dims,
      output_dims,
      stride,
      padding,
      channel_multiplier);
  const data_dims_t input{input_dims[0], input_dims[1], input_dims[2], 0};
  const data_dims_t filter{filter_dims[0], filter_dims[1], filter_dims[2], 0};
  const data_dims_t output{output_dims[0], output_dims[1], output_dims[2], 0};
  dw_conv_params_t params{};
  params.ch_mult = channel_multiplier;
  params.stride = {stride[0], stride[1]};
  params.padding = {padding[0], padding[1]};
  params.dilation = {1, 1};
  return esp_nn_get_depthwise_conv_scratch_size(
      &input, &filter, &output, &params);
}

int32_t softmax_scratch_size(int32_t width, int32_t height) {
  check_bounds(
      std::array{width, height},
      "softmax dimensions",
      1,
      std::numeric_limits<int32_t>::max());
  checked_product({width, sizeof(int32_t)});
  return esp_nn_get_softmax_scratch_size(width, height);
}

int32_t mean_scratch_size(int32_t height, int32_t width, int32_t channels) {
  check_bounds(
      std::array{height, width, channels},
      "mean dimensions",
      1,
      std::numeric_limits<int32_t>::max());
  return esp_nn_get_mean_scratch_size(height, width, channels);
}

int fc_batch_scratch_size(int32_t batches, int32_t row_len, int32_t out_ch) {
  check_bounds(
      std::array{batches}, "batches", 1, std::numeric_limits<int32_t>::max());
  check_bounds(std::array{row_len}, "row_len");
  check_bounds(std::array{out_ch}, "out_ch");
  if (batches <= std::numeric_limits<uint16_t>::max()) {
    check_conv_arithmetic(
        {1, batches, row_len},
        {1, 1, row_len},
        {1, batches, out_ch},
        {1, 1},
        {0, 0});
  }
  return esp_nn_get_fc_batch_scratch_size(
      batches, static_cast<uint16_t>(row_len), static_cast<uint16_t>(out_ch));
}

} // namespace

namespace py = pybind11;

PYBIND11_MODULE(ESP_NN_SCRATCH_MODULE, m) {
  m.doc() =
      "ESP-NN scratch sizes in bytes. Dimensions use (width, height, "
      "channels); stride and padding use (width, height).";
  m.def(
      "conv2d_scratch_size",
      &conv2d_scratch_size,
      py::arg("input_dims"),
      py::arg("filter_dims"),
      py::arg("output_dims"),
      py::arg("stride"),
      py::arg("padding"));
  m.def(
      "depthwise_conv2d_scratch_size",
      &depthwise_conv2d_scratch_size,
      py::arg("input_dims"),
      py::arg("filter_dims"),
      py::arg("output_dims"),
      py::arg("stride"),
      py::arg("padding"),
      py::arg("channel_multiplier"));
  m.def(
      "softmax_scratch_size",
      &softmax_scratch_size,
      py::arg("width"),
      py::arg("height"));
  m.def(
      "mean_scratch_size",
      &mean_scratch_size,
      py::arg("height"),
      py::arg("width"),
      py::arg("channels"));
  m.def("hard_swish_scratch_size", []() {
    return esp_nn_get_hard_swish_scratch_size();
  });
  m.def(
      "fc_batch_scratch_size",
      &fc_batch_scratch_size,
      py::arg("batches"),
      py::arg("row_len"),
      py::arg("out_ch"));
  m.def("logistic_scratch_size", &esp_nn_get_logistic_s8_scratch_size);
}
