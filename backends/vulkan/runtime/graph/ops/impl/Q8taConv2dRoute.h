/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace vkcompute {

struct Q8taConv2dRouteParams final {
  bool is_mali;
  bool supports_int8_dot_product;
  // Adapter::max_buffer_numel(), which returns maxStorageBufferRange in bytes
  // (not elements).
  uint64_t max_buffer_bytes;
  int64_t batch;
  int64_t groups;
  int64_t in_channels_per_group;
  int64_t out_channels;
  int64_t kernel_height;
  int64_t kernel_width;
  int64_t out_height;
  int64_t out_width;
};

bool should_use_q8ta_conv2d_im2col(const Q8taConv2dRouteParams& params);

} // namespace vkcompute
