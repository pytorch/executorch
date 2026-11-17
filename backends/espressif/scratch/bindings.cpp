/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cstdint>

#include <esp_nn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

int conv2d_scratch_size(
    std::array<int32_t, 3> input_dims,
    std::array<int32_t, 3> filter_dims,
    std::array<int32_t, 3> output_dims,
    std::array<int32_t, 2> stride,
    std::array<int32_t, 2> padding) {
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
      &esp_nn_get_softmax_scratch_size,
      py::arg("width"),
      py::arg("height"));
  m.def(
      "mean_scratch_size",
      &esp_nn_get_mean_scratch_size,
      py::arg("height"),
      py::arg("width"),
      py::arg("channels"));
  m.def("hard_swish_scratch_size", []() {
    return esp_nn_get_hard_swish_scratch_size();
  });
  m.def(
      "fc_batch_scratch_size",
      &esp_nn_get_fc_batch_scratch_size,
      py::arg("batches"),
      py::arg("row_len"),
      py::arg("out_ch"));
  m.def("logistic_scratch_size", &esp_nn_get_logistic_s8_scratch_size);
}
