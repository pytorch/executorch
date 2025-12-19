#pragma once

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <cstring>
#include <limits>

STANDALONE_CLANG_DIAGNOSTIC_PUSH()
#if STANDALONE_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
STANDALONE_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

#define EXP_WIDTH_FP8 5
#define MAN_WIDTH_FP8 2
#define EXP_BIAS_FP8 15

namespace standalone::c10 {

/// Constructors

inline STANDALONE_HOST_DEVICE Float8_e5m2::Float8_e5m2(float value)
    : x(detail::fp8e5m2_from_fp32_value(value)) {}

/// Implicit conversions

inline STANDALONE_HOST_DEVICE Float8_e5m2::operator float() const {
  return detail::fp8e5m2_to_fp32_value(x);
}

/// Special values helpers

inline STANDALONE_HOST_DEVICE bool Float8_e5m2::isnan() const {
  return (x & 0b01111111) > 0b01111100;
}

inline STANDALONE_HOST_DEVICE bool Float8_e5m2::isinf() const {
  return (x & 0b01111111) == 0b01111100;
}

/// Arithmetic

inline STANDALONE_HOST_DEVICE Float8_e5m2
operator+(const Float8_e5m2& a, const Float8_e5m2& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e5m2
operator-(const Float8_e5m2& a, const Float8_e5m2& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e5m2
operator*(const Float8_e5m2& a, const Float8_e5m2& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e5m2 operator/(
    const Float8_e5m2& a,
    const Float8_e5m2& b) __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e5m2 operator-(const Float8_e5m2& a) {
  return -static_cast<float>(a);
}

inline STANDALONE_HOST_DEVICE Float8_e5m2& operator+=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a + b;
  return a;
}

inline STANDALONE_HOST_DEVICE Float8_e5m2& operator-=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a - b;
  return a;
}

inline STANDALONE_HOST_DEVICE Float8_e5m2& operator*=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a * b;
  return a;
}

inline STANDALONE_HOST_DEVICE Float8_e5m2& operator/=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline STANDALONE_HOST_DEVICE float operator+(Float8_e5m2 a, float b) {
  return static_cast<float>(a) + b;
}
inline STANDALONE_HOST_DEVICE float operator-(Float8_e5m2 a, float b) {
  return static_cast<float>(a) - b;
}
inline STANDALONE_HOST_DEVICE float operator*(Float8_e5m2 a, float b) {
  return static_cast<float>(a) * b;
}
inline STANDALONE_HOST_DEVICE float operator/(Float8_e5m2 a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline STANDALONE_HOST_DEVICE float operator+(float a, Float8_e5m2 b) {
  return a + static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float operator-(float a, Float8_e5m2 b) {
  return a - static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float operator*(float a, Float8_e5m2 b) {
  return a * static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float operator/(float a, Float8_e5m2 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE float& operator+=(
    float& a,
    const Float8_e5m2& b) {
  return a += static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float& operator-=(
    float& a,
    const Float8_e5m2& b) {
  return a -= static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float& operator*=(
    float& a,
    const Float8_e5m2& b) {
  return a *= static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float& operator/=(
    float& a,
    const Float8_e5m2& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline STANDALONE_HOST_DEVICE double operator+(Float8_e5m2 a, double b) {
  return static_cast<double>(a) + b;
}
inline STANDALONE_HOST_DEVICE double operator-(Float8_e5m2 a, double b) {
  return static_cast<double>(a) - b;
}
inline STANDALONE_HOST_DEVICE double operator*(Float8_e5m2 a, double b) {
  return static_cast<double>(a) * b;
}
inline STANDALONE_HOST_DEVICE double operator/(Float8_e5m2 a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline STANDALONE_HOST_DEVICE double operator+(double a, Float8_e5m2 b) {
  return a + static_cast<double>(b);
}
inline STANDALONE_HOST_DEVICE double operator-(double a, Float8_e5m2 b) {
  return a - static_cast<double>(b);
}
inline STANDALONE_HOST_DEVICE double operator*(double a, Float8_e5m2 b) {
  return a * static_cast<double>(b);
}
inline STANDALONE_HOST_DEVICE double operator/(double a, Float8_e5m2 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline STANDALONE_HOST_DEVICE Float8_e5m2 operator+(Float8_e5m2 a, int b) {
  return a + static_cast<Float8_e5m2>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator-(Float8_e5m2 a, int b) {
  return a - static_cast<Float8_e5m2>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator*(Float8_e5m2 a, int b) {
  return a * static_cast<Float8_e5m2>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator/(Float8_e5m2 a, int b) {
  return a / static_cast<Float8_e5m2>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e5m2 operator+(int a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) + b;
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator-(int a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) - b;
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator*(int a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) * b;
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator/(int a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) / b;
}

//// Arithmetic with int64_t

inline STANDALONE_HOST_DEVICE Float8_e5m2 operator+(Float8_e5m2 a, int64_t b) {
  return a + static_cast<Float8_e5m2>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator-(Float8_e5m2 a, int64_t b) {
  return a - static_cast<Float8_e5m2>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator*(Float8_e5m2 a, int64_t b) {
  return a * static_cast<Float8_e5m2>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator/(Float8_e5m2 a, int64_t b) {
  return a / static_cast<Float8_e5m2>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e5m2 operator+(int64_t a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) + b;
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator-(int64_t a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) - b;
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator*(int64_t a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) * b;
}
inline STANDALONE_HOST_DEVICE Float8_e5m2 operator/(int64_t a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from standalone::c10::Float8_e5m2 to float.

} // namespace standalone::c10

namespace std {

template <>
class numeric_limits<standalone::c10::Float8_e5m2> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_specialized = true;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 3;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 2;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr standalone::c10::Float8_e5m2 min() {
    return standalone::c10::Float8_e5m2(
        0x4, standalone::c10::Float8_e5m2::from_bits());
  }
  static constexpr standalone::c10::Float8_e5m2 max() {
    return standalone::c10::Float8_e5m2(
        0x7B, standalone::c10::Float8_e5m2::from_bits());
  }
  static constexpr standalone::c10::Float8_e5m2 lowest() {
    return standalone::c10::Float8_e5m2(
        0xFB, standalone::c10::Float8_e5m2::from_bits());
  }
  static constexpr standalone::c10::Float8_e5m2 epsilon() {
    return standalone::c10::Float8_e5m2(
        0x34, standalone::c10::Float8_e5m2::from_bits());
  }
  static constexpr standalone::c10::Float8_e5m2 round_error() {
    return standalone::c10::Float8_e5m2(
        0x38, standalone::c10::Float8_e5m2::from_bits());
  }
  static constexpr standalone::c10::Float8_e5m2 infinity() {
    return standalone::c10::Float8_e5m2(
        0x7C, standalone::c10::Float8_e5m2::from_bits());
  }
  static constexpr standalone::c10::Float8_e5m2 quiet_NaN() {
    return standalone::c10::Float8_e5m2(
        0x7F, standalone::c10::Float8_e5m2::from_bits());
  }
  static constexpr standalone::c10::Float8_e5m2 denorm_min() {
    return standalone::c10::Float8_e5m2(
        0x01, standalone::c10::Float8_e5m2::from_bits());
  }
};

} // namespace std

STANDALONE_CLANG_DIAGNOSTIC_POP()
