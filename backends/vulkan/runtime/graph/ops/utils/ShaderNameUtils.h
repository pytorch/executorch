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

#include <sstream>

namespace at {
namespace native {
namespace vulkan {

void apply_dtype_suffix(std::stringstream& kernel_name, const vTensor& tensor);

void apply_memory_layout_suffix(
    std::stringstream& kernel_name,
    const vTensor& tensor);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
