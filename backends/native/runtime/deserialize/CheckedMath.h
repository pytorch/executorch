// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <limits>
#include <type_traits>

namespace ptn::detail {

template <typename T>
constexpr bool checked_add(T lhs, T rhs, T& result) noexcept {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
  if (lhs > std::numeric_limits<T>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

template <typename T>
constexpr bool checked_mul(T lhs, T rhs, T& result) noexcept {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
  if (lhs != 0 && rhs > std::numeric_limits<T>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

} // namespace ptn::detail
