// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Force-included shim for toolchains whose libc++ disables std::random_device.
//
// Arm Toolchain for Embedded ships libc++ with `_LIBCPP_HAS_RANDOM_DEVICE 0`
// (both the picolibc and LLVM-libc variants), so <random> declares but does not
// define std::random_device -- bare metal has no entropy source. The portable
// RNG ops (op_rand / op_randn / op_native_dropout) use it only to seed an
// std::mt19937. Provide a minimal deterministic fallback so they compile.
//
// Inactive on GCC (libstdc++) and AC6 (Arm C++ library): the guard below is
// false there, so the real std::random_device is used unchanged.
#pragma once

#include <random> // pulls in libc++ __config_site -> defines _LIBCPP_HAS_RANDOM_DEVICE

#if defined(_LIBCPP_HAS_RANDOM_DEVICE) && (_LIBCPP_HAS_RANDOM_DEVICE == 0)
namespace std {
class random_device {
 public:
  using result_type = unsigned int;
  random_device() noexcept {}
  // Bare metal has no entropy: return a fixed value (deterministic seed).
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
