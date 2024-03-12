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

namespace at {
namespace native {
namespace vulkan {

api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group);

api::utils::ivec4 get_size_as_ivec4(const vTensor& t);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
