/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>

namespace at {
namespace native {
namespace vulkan {

struct GraphConfig final {
  api::ContextConfig contextConfig;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
