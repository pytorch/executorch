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

// Helper function to get default quant_min and quant_max based on dtype
// This matches the logic in _get_and_check_qmin_qmax from quant_primitives.py
inline std::pair<int, int> get_dtype_bounds(vkapi::ScalarType dtype) {
  switch (dtype) {
    case vkapi::kByte: // uint8
      return {0, 255};
    case vkapi::kChar: // int8
      return {-128, 127};
    case vkapi::kShort: // int16
      return {-(1 << 15), (1 << 15) - 1};
    case vkapi::kInt: // int32
      return {-(1LL << 31), (1LL << 31) - 1};
    default:
      // For unsupported types, throw an error instead of assuming int8
      VK_THROW("Unsupported dtype for quantization bounds: ", dtype);
  }
}

} // namespace vkcompute
