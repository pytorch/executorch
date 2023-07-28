/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/kernels/optimized/utils/llvmMathExtras.h>

namespace executorch {
namespace utils {

template <typename scalar_t>
struct ComputeDTypeTraits {
  using type = scalar_t;
};
// For 16 bit int types, ops should perform internal math in int32_t.
template <>
struct ComputeDTypeTraits<uint16_t> {
  using type = uint32_t;
};
template <>
struct ComputeDTypeTraits<int16_t> {
  using type = int32_t;
};
// For 8 bit int types, ops should perform internal math in int32_t.
template <>
struct ComputeDTypeTraits<uint8_t> {
  using type = uint32_t;
};
template <>
struct ComputeDTypeTraits<int8_t> {
  using type = int32_t;
};

template <typename T>
using compute_dtype = typename ComputeDTypeTraits<T>::type;

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename T>
T CeilLog2(const T& x) {
  if (x <= 2) {
    return 1;
  }
  // Last set bit is floor(log2(x)), floor + 1 is ceil
  // except when x is an exact powers of 2, so subtract 1 first
  return static_cast<T>(executorch::llvm::findLastSet(
      static_cast<uint64_t>(x) - 1)) + 1;
}

} // namespace utils
} // namespace executorch
