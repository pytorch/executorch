/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <optional>
#include <ostream>
#include <vector>

namespace vkcompute {

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << '[';
  for (const auto& elem : vec) {
    os << elem << ',';
  }
  os << ']';
  return os; // Return the ostream to allow chaining
}

inline std::ostream& operator<<(std::ostream& os, const utils::uvec3& v) {
  return utils::operator<<(os, v);
}

inline std::ostream& operator<<(std::ostream& os, const utils::uvec4& v) {
  return utils::operator<<(os, v);
}

inline std::ostream& operator<<(std::ostream& os, const utils::ivec3& v) {
  return utils::operator<<(os, v);
}

inline std::ostream& operator<<(std::ostream& os, const utils::ivec4& v) {
  return utils::operator<<(os, v);
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::optional<T>& opt) {
  os << "[";
  if (opt) {
    os << opt.value();
  }
  os << "]";
  return os;
}

} // namespace vkcompute
