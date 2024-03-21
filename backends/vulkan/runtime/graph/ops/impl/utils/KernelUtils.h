/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

namespace at {
namespace native {
namespace vulkan {

struct KernelParams final {
  api::utils::ivec2 kernel;
  api::utils::ivec2 stride;
  api::utils::ivec2 padding;
  api::utils::ivec2 dilation;
};

int64_t calc_out_size(
    const int64_t in_size,
    const int64_t kernel,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation,
    const bool ceil_mode);

api::utils::ivec2 normalize_wh(Value& v);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
