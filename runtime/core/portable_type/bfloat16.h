/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>

namespace executorch {
namespace runtime {
namespace etensor {

namespace internal {
inline float f32_from_bits(uint16_t src) {
  float res = 0;
  uint32_t tmp = src;
  tmp <<= 16;
  std::memcpy(&res, &tmp, sizeof(tmp));
  return res;
}

inline uint16_t round_to_nearest_even(float src) {
  if (std::isnan(src)) {
    return UINT16_C(0x7FC0);
  }
  uint32_t U32 = 0;
  std::memcpy(&U32, &src, sizeof(U32));
  uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
  return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
}
} // namespace internal

/**
 * The "brain floating-point" type, compatible with c10/util/BFloat16.h from
 * pytorch core.
 *
 * This representation uses 1 bit for the sign, 8 bits for the exponent and 7
 * bits for the mantissa.
 */
struct alignas(2) BFloat16 {
  uint16_t x;

  BFloat16() = default;
  struct from_bits_t {};
  static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr BFloat16(unsigned short bits, from_bits_t) : x(bits) {}
  /* implicit */ BFloat16(float value)
      : x(internal::round_to_nearest_even(value)) {}
  operator float() const {
    return internal::f32_from_bits(x);
  }
};

inline std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  out << (float)value;
  return out;
}

/// Arithmetic

inline BFloat16 operator+(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline BFloat16 operator-(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline BFloat16 operator*(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline BFloat16 operator/(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline BFloat16 operator-(const BFloat16& a) {
  return -static_cast<float>(a);
}

inline BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
  a = a + b;
  return a;
}

inline BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
  a = a - b;
  return a;
}

inline BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
  a = a * b;
  return a;
}

inline BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
  a = a / b;
  return a;
}

inline BFloat16& operator|(BFloat16& a, const BFloat16& b) {
  a.x = a.x | b.x;
  return a;
}

inline BFloat16& operator^(BFloat16& a, const BFloat16& b) {
  a.x = a.x ^ b.x;
  return a;
}

inline BFloat16& operator&(BFloat16& a, const BFloat16& b) {
  a.x = a.x & b.x;
  return a;
}

/// Arithmetic with floats

inline float operator+(BFloat16 a, float b) {
  return static_cast<float>(a) + b;
}
inline float operator-(BFloat16 a, float b) {
  return static_cast<float>(a) - b;
}
inline float operator*(BFloat16 a, float b) {
  return static_cast<float>(a) * b;
}
inline float operator/(BFloat16 a, float b) {
  return static_cast<float>(a) / b;
}

inline float operator+(float a, BFloat16 b) {
  return a + static_cast<float>(b);
}
inline float operator-(float a, BFloat16 b) {
  return a - static_cast<float>(b);
}
inline float operator*(float a, BFloat16 b) {
  return a * static_cast<float>(b);
}
inline float operator/(float a, BFloat16 b) {
  return a / static_cast<float>(b);
}

inline float& operator+=(float& a, const BFloat16& b) {
  return a += static_cast<float>(b);
}
inline float& operator-=(float& a, const BFloat16& b) {
  return a -= static_cast<float>(b);
}
inline float& operator*=(float& a, const BFloat16& b) {
  return a *= static_cast<float>(b);
}
inline float& operator/=(float& a, const BFloat16& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline double operator+(BFloat16 a, double b) {
  return static_cast<double>(a) + b;
}
inline double operator-(BFloat16 a, double b) {
  return static_cast<double>(a) - b;
}
inline double operator*(BFloat16 a, double b) {
  return static_cast<double>(a) * b;
}
inline double operator/(BFloat16 a, double b) {
  return static_cast<double>(a) / b;
}

inline double operator+(double a, BFloat16 b) {
  return a + static_cast<double>(b);
}
inline double operator-(double a, BFloat16 b) {
  return a - static_cast<double>(b);
}
inline double operator*(double a, BFloat16 b) {
  return a * static_cast<double>(b);
}
inline double operator/(double a, BFloat16 b) {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline BFloat16 operator+(BFloat16 a, int b) {
  return a + static_cast<BFloat16>(b);
}
inline BFloat16 operator-(BFloat16 a, int b) {
  return a - static_cast<BFloat16>(b);
}
inline BFloat16 operator*(BFloat16 a, int b) {
  return a * static_cast<BFloat16>(b);
}
inline BFloat16 operator/(BFloat16 a, int b) {
  return a / static_cast<BFloat16>(b);
}

inline BFloat16 operator+(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) + b;
}
inline BFloat16 operator-(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) - b;
}
inline BFloat16 operator*(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) * b;
}
inline BFloat16 operator/(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) / b;
}

//// Arithmetic with int64_t

inline BFloat16 operator+(BFloat16 a, int64_t b) {
  return a + static_cast<BFloat16>(b);
}
inline BFloat16 operator-(BFloat16 a, int64_t b) {
  return a - static_cast<BFloat16>(b);
}
inline BFloat16 operator*(BFloat16 a, int64_t b) {
  return a * static_cast<BFloat16>(b);
}
inline BFloat16 operator/(BFloat16 a, int64_t b) {
  return a / static_cast<BFloat16>(b);
}

inline BFloat16 operator+(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) + b;
}
inline BFloat16 operator-(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) - b;
}
inline BFloat16 operator*(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) * b;
}
inline BFloat16 operator/(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) / b;
}

// Overloading < and > operators, because std::max and std::min use them.

inline bool operator>(BFloat16& lhs, BFloat16& rhs) {
  return float(lhs) > float(rhs);
}

inline bool operator<(BFloat16& lhs, BFloat16& rhs) {
  return float(lhs) < float(rhs);
}

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::BFloat16;
} // namespace executor
} // namespace torch

namespace std {

template <>
class numeric_limits<executorch::runtime::etensor::BFloat16> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr torch::executor::BFloat16 min() {
    return torch::executor::BFloat16(
        0x0080, torch::executor::BFloat16::from_bits());
  }
  static constexpr torch::executor::BFloat16 lowest() {
    return torch::executor::BFloat16(
        0xFF7F, torch::executor::BFloat16::from_bits());
  }
  static constexpr torch::executor::BFloat16 max() {
    return torch::executor::BFloat16(
        0x7F7F, torch::executor::BFloat16::from_bits());
  }
  static constexpr torch::executor::BFloat16 epsilon() {
    return torch::executor::BFloat16(
        0x3C00, torch::executor::BFloat16::from_bits());
  }
  static constexpr torch::executor::BFloat16 round_error() {
    return torch::executor::BFloat16(
        0x3F00, torch::executor::BFloat16::from_bits());
  }
  static constexpr torch::executor::BFloat16 infinity() {
    return torch::executor::BFloat16(
        0x7F80, torch::executor::BFloat16::from_bits());
  }
  static constexpr torch::executor::BFloat16 quiet_NaN() {
    return torch::executor::BFloat16(
        0x7FC0, torch::executor::BFloat16::from_bits());
  }
  static constexpr torch::executor::BFloat16 signaling_NaN() {
    return torch::executor::BFloat16(
        0x7F80, torch::executor::BFloat16::from_bits());
  }
  static constexpr torch::executor::BFloat16 denorm_min() {
    return torch::executor::BFloat16(
        0x0001, torch::executor::BFloat16::from_bits());
  }
};

} // namespace std
