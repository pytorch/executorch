/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2024,2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/util/irange.h>
#include <executorch/runtime/platform/compiler.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <ostream>
#include <type_traits>
/**
 * @file
 * This header defines common, low-level operations that can often be
 * vectorized/accelerated on hardware targets.
 *
 * Although they do not yet have hardware-optimized implementations, operators
 * that use this API can benefit from optimizations in the future.
 */

namespace torch {
namespace executor {

/// Returns the minimum element of the array at `x`, which must have `size`
/// elements.
inline float vec_minf(const float* x, size_t size) {
  return *std::min_element(x, x + size);
}

/// Returns the maximum element of the array at `x`, which must have `size`
/// elements.
inline float vec_maxf(const float* x, size_t size) {
  return *std::max_element(x, x + size);
}

/// Add each element of `x` and `y` into the corresponding element of `z`. All
/// arrays must have `size` elements.
inline void vec_addf(
    float* ET_RESTRICT z,
    const float* ET_RESTRICT x,
    const float* ET_RESTRICT y,
    size_t size) {
  for (const auto i : c10::irange(size)) {
    z[i] = x[i] + y[i];
  }
}

/// Multiplies every element of `x` by `scale`, and writes the result into the
/// corresponding element of `y`. `x` and `y` must have `size` elements.
inline void vec_scalef(
    float* ET_RESTRICT y,
    const float* ET_RESTRICT x,
    float scale,
    size_t size) {
  for (const auto i : c10::irange(size)) {
    y[i] = x[i] * scale;
  }
}

/// x: m * n, y: n * p, z: m * p.
/// z[i][j] = sum(x[i][k] * y[k][j])
template <typename T, typename U = T>
inline void vec_matmul(
    T* ET_RESTRICT z,
    const U* ET_RESTRICT x,
    const U* ET_RESTRICT y,
    int64_t m,
    int64_t n,
    int64_t p) {
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(p)) {
      T sum = 0;
      for (const auto k : c10::irange(n)) {
        sum += x[i * n + k] * y[k * p + j];
      }
      z[i * p + j] = sum;
    }
  }
}

template <typename T, typename U = T>
inline void vec_quantized_matmul_int8(
    T* ET_RESTRICT z,
    const U* ET_RESTRICT x,
    const int8_t* ET_RESTRICT y,
    const U* ET_RESTRICT s,
    int64_t m,
    int64_t n,
    int64_t p) {
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(p)) {
      T sum = 0;
      for (const auto k : c10::irange(n)) {
        sum += x[i * n + k] * static_cast<U>(y[k * p + j]) * s[k];
      }
      z[i * p + j] = sum;
    }
  }
}

static inline size_t bounds_min(size_t a, size_t b) {
  return (a < b) ? a : b;
}

/// x: m * n, y: p * n, z: m * p, s: p * groups
/// z[i][j] = sum(x[i][k] * y[j][k] * s[j][k/g])
template <typename T, typename U = T, typename V = U>
inline void vec_quantized_matmul_transb_int8(
    T* ET_RESTRICT z,
    const U* ET_RESTRICT x,
    const int8_t* ET_RESTRICT y,
    const V* ET_RESTRICT s,
    int64_t m,
    int64_t n,
    int64_t p,
    int64_t g) {
  int64_t n_over_g = (n + g - 1) / g;

  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(p)) {
      T sum = 0;
      for (int64_t k = 0; k < n; k += g) {
        T psum = 0;
        // the last group may have fewer than g elements
        for (const auto k2 : c10::irange(k, bounds_min(k + g, n))) {
          psum += x[i * n + k2] * static_cast<U>(y[j * n + k2]);
        }
        sum += psum * s[j * n_over_g + k / g];
      }
      z[i * p + j] = sum;
    }
  }
}

// mat1 (m x n), mat2 (n x p), out (m, p), self (m x p)
// z[i][j] = sum(x[i][k] * y[k][j]), for k in range(n)
// T for tensor dtype, U for scalar type
template <typename T, typename U = T>
inline void vec_addmm(
    T* ET_RESTRICT out_data,
    const T* ET_RESTRICT self_data,
    const T* ET_RESTRICT mat1_data,
    const T* ET_RESTRICT mat2_data,
    int64_t m,
    int64_t n,
    int64_t p,
    U beta,
    U alpha) {
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(p)) {
      T sum = 0;
      for (const auto k : c10::irange(n)) {
        sum += mat1_data[i * n + k] * mat2_data[k * p + j];
      }
      out_data[i * p + j] = sum * alpha + self_data[i * p + j] * beta;
    }
  }
}

/// Returns the sum of all elements in `x`, which must have `size` elements.
template <typename T>
inline float reduce_add(const T* x, size_t size) {
  return std::accumulate(x, x + size, 0.f);
}

/// Returns the sum of the squares of all elements in `x`, which must have
/// `size` elements.
template <typename T>
inline float vec_powerf(const T* x, size_t size) {
  float sum = 0;
  for (const auto i : c10::irange(size)) {
    sum += static_cast<float>(x[i]) * x[i];
  }
  return sum;
}

/// Computes the result of softmax(x, x+n), write into y.
/// y = e ^ (x - max(x)) / sum(e^(x - max(x))
/// T, U can only be one of double, float
template <
    typename T,
    typename U,
    typename checkT = typename std::enable_if<
        std::is_same<float, typename std::remove_cv<T>::type>::value ||
        std::is_same<double, typename std::remove_cv<T>::type>::value>::type,
    typename checkU = typename std::enable_if<
        std::is_same<float, typename std::remove_cv<U>::type>::value ||
        std::is_same<double, typename std::remove_cv<U>::type>::value>::type>
inline void vec_softmax(T* ET_RESTRICT y, const U* ET_RESTRICT x, int n) {
  U max_x = *std::max_element(x, x + n);
  T sum = 0;

  for (const auto i : c10::irange(n)) {
    y[i] = std::exp(x[i] - max_x);
    sum += y[i];
  }

  for (const auto i : c10::irange(n)) {
    y[i] /= sum;
  }
}

namespace internal {
template <class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
#ifdef __cpp_lib_clamp
  return std::clamp(v, lo, hi);
#else
  return v < lo ? lo : hi < v ? hi : v;
#endif
}
} // namespace internal

/// Quantizes the elements of `x` into `y`, both of which must have `size`
/// elements. Inverse of `dequantize_i8_f32()`.
inline void quantize_i8_f32(
    int8_t* ET_RESTRICT y,
    const float* ET_RESTRICT x,
    float scale,
    int32_t zero_point,
    size_t size) {
  for (const auto i : c10::irange(size)) {
    float tmp = std::round(x[i] * scale + zero_point);
    y[i] = internal::clamp(tmp, -128.f, 127.f);
  }
}

/// Dequantizes the elements of `x` into `y`, both of which must have `size`
/// elements. Inverse of `quantize_i8_f32()`.
inline void dequantize_i8_f32(
    float* ET_RESTRICT y,
    const int8_t* ET_RESTRICT x,
    float scale,
    int32_t zero_point,
    size_t size) {
  for (const auto i : c10::irange(size)) {
    y[i] = scale * (x[i] - zero_point);
  }
}

} // namespace executor
} // namespace torch
