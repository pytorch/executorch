#pragma once

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <cstdint>
#include <limits>

STANDALONE_CLANG_DIAGNOSTIC_PUSH()
#if STANDALONE_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
STANDALONE_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace standalone::c10 {

/// Constructors

inline STANDALONE_HOST_DEVICE Float8_e4m3fn::Float8_e4m3fn(float value)
    : x(detail::fp8e4m3fn_from_fp32_value(value)) {}

/// Implicit conversions

inline STANDALONE_HOST_DEVICE Float8_e4m3fn::operator float() const {
  return detail::fp8e4m3fn_to_fp32_value(x);
}

/// Special values helper

inline STANDALONE_HOST_DEVICE bool Float8_e4m3fn::isnan() const {
  return (x & 0b01111111) == 0b01111111;
}

/// Arithmetic

inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator+(const Float8_e4m3fn& a, const Float8_e4m3fn& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator-(const Float8_e4m3fn& a, const Float8_e4m3fn& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator*(const Float8_e4m3fn& a, const Float8_e4m3fn& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator/(
    const Float8_e4m3fn& a,
    const Float8_e4m3fn& b) __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator-(const Float8_e4m3fn& a) {
  return -static_cast<float>(a);
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn& operator+=(
    Float8_e4m3fn& a,
    const Float8_e4m3fn& b) {
  a = a + b;
  return a;
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn& operator-=(
    Float8_e4m3fn& a,
    const Float8_e4m3fn& b) {
  a = a - b;
  return a;
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn& operator*=(
    Float8_e4m3fn& a,
    const Float8_e4m3fn& b) {
  a = a * b;
  return a;
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn& operator/=(
    Float8_e4m3fn& a,
    const Float8_e4m3fn& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline STANDALONE_HOST_DEVICE float operator+(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) + b;
}
inline STANDALONE_HOST_DEVICE float operator-(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) - b;
}
inline STANDALONE_HOST_DEVICE float operator*(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) * b;
}
inline STANDALONE_HOST_DEVICE float operator/(Float8_e4m3fn a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline STANDALONE_HOST_DEVICE float operator+(float a, Float8_e4m3fn b) {
  return a + static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float operator-(float a, Float8_e4m3fn b) {
  return a - static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float operator*(float a, Float8_e4m3fn b) {
  return a * static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float operator/(float a, Float8_e4m3fn b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline STANDALONE_HOST_DEVICE float& operator+=(
    float& a,
    const Float8_e4m3fn& b) {
  return a += static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float& operator-=(
    float& a,
    const Float8_e4m3fn& b) {
  return a -= static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float& operator*=(
    float& a,
    const Float8_e4m3fn& b) {
  return a *= static_cast<float>(b);
}
inline STANDALONE_HOST_DEVICE float& operator/=(
    float& a,
    const Float8_e4m3fn& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline STANDALONE_HOST_DEVICE double operator+(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) + b;
}
inline STANDALONE_HOST_DEVICE double operator-(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) - b;
}
inline STANDALONE_HOST_DEVICE double operator*(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) * b;
}
inline STANDALONE_HOST_DEVICE double operator/(Float8_e4m3fn a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline STANDALONE_HOST_DEVICE double operator+(double a, Float8_e4m3fn b) {
  return a + static_cast<double>(b);
}
inline STANDALONE_HOST_DEVICE double operator-(double a, Float8_e4m3fn b) {
  return a - static_cast<double>(b);
}
inline STANDALONE_HOST_DEVICE double operator*(double a, Float8_e4m3fn b) {
  return a * static_cast<double>(b);
}
inline STANDALONE_HOST_DEVICE double operator/(double a, Float8_e4m3fn b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator+(Float8_e4m3fn a, int b) {
  return a + static_cast<Float8_e4m3fn>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator-(Float8_e4m3fn a, int b) {
  return a - static_cast<Float8_e4m3fn>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator*(Float8_e4m3fn a, int b) {
  return a * static_cast<Float8_e4m3fn>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator/(Float8_e4m3fn a, int b) {
  return a / static_cast<Float8_e4m3fn>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator+(int a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) + b;
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator-(int a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) - b;
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator*(int a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) * b;
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn operator/(int a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) / b;
}

//// Arithmetic with int64_t

inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator+(Float8_e4m3fn a, int64_t b) {
  return a + static_cast<Float8_e4m3fn>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator-(Float8_e4m3fn a, int64_t b) {
  return a - static_cast<Float8_e4m3fn>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator*(Float8_e4m3fn a, int64_t b) {
  return a * static_cast<Float8_e4m3fn>(b);
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator/(Float8_e4m3fn a, int64_t b) {
  return a / static_cast<Float8_e4m3fn>(b);
}

inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator+(int64_t a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) + b;
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator-(int64_t a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) - b;
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator*(int64_t a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) * b;
}
inline STANDALONE_HOST_DEVICE Float8_e4m3fn
operator/(int64_t a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from standalone::c10::Float8_e4m3fn to float.

} // namespace standalone::c10

namespace std {

template <>
class numeric_limits<standalone::c10::Float8_e4m3fn> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 4;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 3;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -5;
  static constexpr int min_exponent10 = -1;
  static constexpr int max_exponent = 8;
  static constexpr int max_exponent10 = 2;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  static constexpr standalone::c10::Float8_e4m3fn min() {
    return standalone::c10::Float8_e4m3fn(
        0x08, standalone::c10::Float8_e4m3fn::from_bits());
  }
  static constexpr standalone::c10::Float8_e4m3fn lowest() {
    return standalone::c10::Float8_e4m3fn(
        0xFE, standalone::c10::Float8_e4m3fn::from_bits());
  }
  static constexpr standalone::c10::Float8_e4m3fn max() {
    return standalone::c10::Float8_e4m3fn(
        0x7E, standalone::c10::Float8_e4m3fn::from_bits());
  }
  static constexpr standalone::c10::Float8_e4m3fn epsilon() {
    return standalone::c10::Float8_e4m3fn(
        0x20, standalone::c10::Float8_e4m3fn::from_bits());
  }
  static constexpr standalone::c10::Float8_e4m3fn round_error() {
    return standalone::c10::Float8_e4m3fn(
        0x30, standalone::c10::Float8_e4m3fn::from_bits());
  }
  static constexpr standalone::c10::Float8_e4m3fn quiet_NaN() {
    return standalone::c10::Float8_e4m3fn(
        0x7F, standalone::c10::Float8_e4m3fn::from_bits());
  }
  static constexpr standalone::c10::Float8_e4m3fn denorm_min() {
    return standalone::c10::Float8_e4m3fn(
        0x01, standalone::c10::Float8_e4m3fn::from_bits());
  }
};

} // namespace std

STANDALONE_CLANG_DIAGNOSTIC_POP()
