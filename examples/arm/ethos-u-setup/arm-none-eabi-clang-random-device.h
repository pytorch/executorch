// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <random>

#if defined(_LIBCPP_HAS_RANDOM_DEVICE) && (_LIBCPP_HAS_RANDOM_DEVICE == 0)
namespace std {
class random_device {
 public:
  using result_type = unsigned int;

  random_device() noexcept {}
  result_type operator()() noexcept {
    return 0x9e3779b9u;
  }
  double entropy() const noexcept {
    return 0.0;
  }
  static constexpr result_type min() noexcept {
    return 0u;
  }
  static constexpr result_type max() noexcept {
    return ~0u;
  }

  random_device(const random_device&) = delete;
  random_device& operator=(const random_device&) = delete;
};
} // namespace std
#endif
