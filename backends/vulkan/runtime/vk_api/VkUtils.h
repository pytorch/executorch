/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

namespace vkcompute {
namespace vkapi {

inline VkExtent3D create_extent3d(const utils::uvec3& extents) {
  return VkExtent3D{extents.data[0u], extents.data[1u], extents.data[2u]};
}

} // namespace vkapi
} // namespace vkcompute
