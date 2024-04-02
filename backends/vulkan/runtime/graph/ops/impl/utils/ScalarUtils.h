/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

namespace vkcompute {

template <typename T>
T extract_scalar(const Value& value) {
  if (value.isInt()) {
    return static_cast<T>(value.toInt());
  }
  if (value.isDouble()) {
    return static_cast<T>(value.toDouble());
  }
  if (value.isBool()) {
    return static_cast<T>(value.toBool());
  }
  VK_THROW("Cannot extract scalar from Value with type ", value.type());
}

} // namespace vkcompute
