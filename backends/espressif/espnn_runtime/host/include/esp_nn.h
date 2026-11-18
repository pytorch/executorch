/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// esp-nn's header. esp-nn's portable kernels need no scratch, unlike its
// ESP32-S3 kernels. With ESPNN_HOST_SCRATCH, the convolution and mean kernels
// request some and fill all of it before they run, so that a scratch buffer
// that is too small fails under AddressSanitizer, and one that overlaps live
// activations fails the bit-exact output checks.

#pragma once

#include_next <esp_nn.h>

#ifdef ESPNN_HOST_SCRATCH

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace espnn_host {

inline void* g_conv_scratch = nullptr;
inline void* g_depthwise_scratch = nullptr;
inline void* g_mean_scratch = nullptr;

// Each term is capped, so that no graph shape can overflow the sizes.
inline int capped(int64_t value) {
  return static_cast<int>(std::min<int64_t>(value, 4096));
}

inline int scratch_size(const data_dims_t* input, const data_dims_t* output) {
  return 4 * capped(input->channels) +
      capped(int64_t{output->width} * output->height) + 16;
}

inline int conv_scratch_size(
    const data_dims_t* input_dims,
    const data_dims_t*,
    const data_dims_t* output_dims,
    const conv_params_t*) {
  return scratch_size(input_dims, output_dims);
}

inline int depthwise_conv_scratch_size(
    const data_dims_t* input_dims,
    const data_dims_t*,
    const data_dims_t* output_dims,
    const dw_conv_params_t*) {
  return scratch_size(input_dims, output_dims);
}

inline int32_t mean_scratch_size(int32_t, int32_t, int32_t channels) {
  return 4 * capped(channels) + 16;
}

inline void set_conv_scratch(const void* buffer) {
  g_conv_scratch = const_cast<void*>(buffer);
}

inline void set_depthwise_conv_scratch(const void* buffer) {
  g_depthwise_scratch = const_cast<void*>(buffer);
}

inline void set_mean_scratch(void* buffer) {
  g_mean_scratch = buffer;
}

inline void conv_s8(
    const data_dims_t* input_dims,
    const int8_t* input_data,
    const data_dims_t* filter_dims,
    const int8_t* filter_data,
    const int32_t* bias,
    const data_dims_t* output_dims,
    int8_t* out_data,
    const conv_params_t* conv_params,
    const quant_data_t* quant_data) {
  std::memset(g_conv_scratch, 0x5a, scratch_size(input_dims, output_dims));
  esp_nn_conv_s8(
      input_dims,
      input_data,
      filter_dims,
      filter_data,
      bias,
      output_dims,
      out_data,
      conv_params,
      quant_data);
}

inline void depthwise_conv_s8(
    const data_dims_t* input_dims,
    const int8_t* input_data,
    const data_dims_t* filter_dims,
    const int8_t* filter_data,
    const int32_t* bias,
    const data_dims_t* output_dims,
    int8_t* out_data,
    const dw_conv_params_t* conv_params,
    const quant_data_t* quant_data) {
  std::memset(g_depthwise_scratch, 0x5a, scratch_size(input_dims, output_dims));
  esp_nn_depthwise_conv_s8(
      input_dims,
      input_data,
      filter_dims,
      filter_data,
      bias,
      output_dims,
      out_data,
      conv_params,
      quant_data);
}

inline void mean_nhwc_s8(
    const int8_t* input,
    int8_t* output,
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t multiplier,
    int32_t shift) {
  std::memset(g_mean_scratch, 0x5a, mean_scratch_size(height, width, channels));
  esp_nn_mean_nhwc_s8(
      input,
      output,
      height,
      width,
      channels,
      input_zero_point,
      output_zero_point,
      multiplier,
      shift);
}

} // namespace espnn_host

#undef esp_nn_get_conv_scratch_size
#undef esp_nn_set_conv_scratch_buf
#undef esp_nn_conv_s8
#undef esp_nn_get_depthwise_conv_scratch_size
#undef esp_nn_set_depthwise_conv_scratch_buf
#undef esp_nn_depthwise_conv_s8
#undef esp_nn_get_mean_scratch_size
#undef esp_nn_set_mean_scratch_buf
#undef esp_nn_mean_nhwc_s8
#define esp_nn_get_conv_scratch_size espnn_host::conv_scratch_size
#define esp_nn_set_conv_scratch_buf espnn_host::set_conv_scratch
#define esp_nn_conv_s8 espnn_host::conv_s8
#define esp_nn_get_depthwise_conv_scratch_size \
  espnn_host::depthwise_conv_scratch_size
#define esp_nn_set_depthwise_conv_scratch_buf \
  espnn_host::set_depthwise_conv_scratch
#define esp_nn_depthwise_conv_s8 espnn_host::depthwise_conv_s8
#define esp_nn_get_mean_scratch_size espnn_host::mean_scratch_size
#define esp_nn_set_mean_scratch_buf espnn_host::set_mean_scratch
#define esp_nn_mean_nhwc_s8 espnn_host::mean_nhwc_s8

#endif
