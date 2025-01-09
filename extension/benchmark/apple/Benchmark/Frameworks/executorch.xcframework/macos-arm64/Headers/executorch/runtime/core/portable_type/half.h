/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <math.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>

#if defined(__GNUC__) || defined(__clang__)
#if defined(__aarch64__)
#ifndef __ARM_V8_ONLY__
#define NATIVE_FP16 1
#endif // __ARM_V8_ONLY__
#endif // __aarch64__
#endif // GNUC or clang

#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
#if defined(__AVX2__)
#define X86_F16 1
#include <immintrin.h> // import conversion ops from f16cintrin.h
#endif // __AVX2__
#endif // __x86_64__ || _M_X64 || __i386 || _M_IX86
#endif // __GNUC__ || __clang__

namespace executorch {
namespace runtime {
namespace etensor {

/**
 * A half-precision floating point type, compatible with c10/util/Half.h from
 * pytorch core.
 */
struct alignas(2) Half {
  union {
#ifdef NATIVE_FP16
    _Float16 y;
#endif
    uint16_t x;
  };

  struct from_bits_t {};
  static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  Half() = default;

  constexpr Half(uint16_t bits, from_bits_t) : x(bits) {}
  /* implicit */ inline Half(float value);
  inline operator float() const;
};

namespace internal {

inline float fp32_from_bits(uint32_t w) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {w};
  return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = {f};
  return fp32.as_bits;
}

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the bits 0-30
   * of the 32-bit word:
   *
   *      +---+-----+------------+-------------------+
   *      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  30  27-31     17-26            0-16
   */
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  /*
   * Renorm shift is the number of bits to shift mantissa left to make the
   * half-precision number normalized. If the initial number is normalized, some
   * of its high 6 bits (sign == 0 and 5-bit exponent) equals one. In this case
   * renorm_shift == 0. If the number is denormalize, renorm_shift > 0. Note
   * that if we shift denormalized nonsign by renorm_shift, the unit bit of
   * mantissa will shift into exponent, turning the biased exponent into 1, and
   * making mantissa normalized (i.e. without leading 1).
   */
#ifdef _MSC_VER
  unsigned long nonsign_bsr;
  _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
  uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
  uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
  renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
  /*
   * Iff half-precision number has exponent of 15, the addition overflows
   * it into bit 31, and the subsequent shift turns the high 9 bits
   * into 1. Thus inf_nan_mask == 0x7F800000 if the half-precision number
   * had exponent of 15 (i.e. was NaN or infinity) 0x00000000 otherwise
   */
  const int32_t inf_nan_mask =
      ((int32_t)(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
  /*
   * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31
   * into 1. Otherwise, bit 31 remains 0. The signed shift right by 31
   * broadcasts bit 31 into all bits of the zero_mask. Thus zero_mask ==
   * 0xFFFFFFFF if the half-precision number was zero (+0.0h or -0.0h)
   * 0x00000000 otherwise
   */
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  /*
   * 1. Shift nonsign left by renorm_shift to normalize it (if the input
   * was denormal)
   * 2. Shift nonsign right by 3 so the exponent (5 bits originally)
   * becomes an 8-bit field and 10-bit mantissa shifts into the 10 high
   * bits of the 23-bit mantissa of IEEE single-precision number.
   * 3. Add 0x70 to the exponent (starting at bit 23) to compensate the
   * different in exponent bias (0x7F for single-precision number less 0xF
   * for half-precision number).
   * 4. Subtract renorm_shift from the exponent (starting at bit 23) to
   * account for renormalization. As renorm_shift is less than 0x70, this
   * can be combined with step 3.
   * 5. Binary OR with inf_nan_mask to turn the exponent into 0xFF if the
   * input was NaN or infinity.
   * 6. Binary ANDNOT with zero_mask to turn the mantissa and exponent
   * into zero if the input was zero.
   * 7. Combine with the sign of the input number.
   */
  return sign |
      ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) |
        inf_nan_mask) &
       ~zero_mask);
}

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
inline float fp16_ieee_to_fp32_value(uint16_t h) {
#ifdef X86_F16
  return _cvtsh_ss(h);
#else

  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, there are some adjustments to the exponent:
   * - The exponent needs to be corrected by the difference in exponent bias
   * between single-precision and half-precision formats (0x7F - 0xF = 0x70)
   * - Inf and NaN values in the inputs should become Inf and NaN values after
   * conversion to the single-precision number. Therefore, if the biased
   * exponent of the half-precision input was 0x1F (max possible value), the
   * biased exponent of the single-precision output must be 0xFF (max possible
   * value). We do this correction in two steps:
   *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset
   * below) rather than by 0x70 suggested by the difference in the exponent bias
   * (see above).
   *   - Then we multiply the single-precision result of exponent adjustment by
   * 2**(-112) to reverse the effect of exponent adjustment by 0xE0 less the
   * necessary exponent adjustment by 0x70 due to difference in exponent bias.
   *     The floating-point multiplication hardware would ensure than Inf and
   * NaN would retain their value on at least partially IEEE754-compliant
   * implementations.
   *
   * Note that the above operations do not handle denormal inputs (where biased
   * exponent == 0). However, they also do not operate on denormal inputs, and
   * do not produce denormal results.
   */
  constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
  // const float exp_scale = 0x1.0p-112f;
  constexpr uint32_t scale_bits = (uint32_t)15 << 23;
  float exp_scale_val = 0;
  std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
  const float exp_scale = exp_scale_val;
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero, and mantissa has
   * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized half-precision numbers are represented as:
   *    FP16 = mantissa * 2**(-24).
   * The trick is to construct a normalized single-precision number with the
   * same mantissa and thehalf-precision input and with an exponent which would
   * scale the corresponding mantissa bits to 2**(-24). A normalized
   * single-precision floating-point number is represented as: FP32 = (1 +
   * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
   * exponent is 126, a unit change in the mantissa of the input denormalized
   * half-precision number causes a change of the constructed single-precision
   * number by 2**(-24), i.e. the same amount.
   *
   * The last step is to adjust the bias of the constructed single-precision
   * number. When the input half-precision number is zero, the constructed
   * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
   * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
   * single-precision number to get the numerical equivalent of the input
   * half-precision number.
   */
  constexpr uint32_t magic_mask = UINT32_C(126) << 23;
  constexpr float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * - Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent. The variable
   * two_w contains input exponent in bits 27-31, therefore if its smaller than
   * 2**27, the input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign
   * of the input number.
   */
  constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
      (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                   : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);

#endif // not X86_F16
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in IEEE half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
inline uint16_t fp16_ieee_from_fp32_value(float f) {
#ifdef X86_F16
  return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else

  // const float scale_to_inf = 0x1.0p+112f;
  // const float scale_to_zero = 0x1.0p-110f;
  constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
  constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;
  float scale_to_inf_val = 0, scale_to_zero_val = 0;
  std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
  std::memcpy(
      &scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
  const float scale_to_inf = scale_to_inf_val;
  const float scale_to_zero = scale_to_zero_val;

#if defined(_MSC_VER) && _MSC_VER == 1916
  float base = ((signbit(f) != 0 ? -f : f) * scale_to_inf) * scale_to_zero;
#else
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
#endif

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
#endif // not X86_F16
}

} // namespace internal

/// Constructors
#ifdef NATIVE_FP16
inline Half::Half(float value) : y(value) {}
#else
inline Half::Half(float value)
    : x(internal::fp16_ieee_from_fp32_value(value)) {}
#endif

/// Implicit conversions
#ifdef NATIVE_FP16
inline Half::operator float() const {
  return (float)y;
}
#else
inline Half::operator float() const {
  return internal::fp16_ieee_to_fp32_value(x);
}
#endif

/// Arithmetic

#ifdef NATIVE_FP16

#define return_half(r) \
  do {                 \
    Half ret;          \
    ret.y = r;         \
    return ret;        \
  } while (0)

inline Half operator+(const Half& a, const Half& b) {
  return_half(a.y + b.y);
}

inline Half operator-(const Half& a, const Half& b) {
  return_half(a.y - b.y);
  return static_cast<float>(a) - static_cast<float>(b);
}

inline Half operator*(const Half& a, const Half& b) {
  return_half(a.y * b.y);
}

inline Half operator/(const Half& a, const Half& b) {
  return_half(a.y / b.y);
}

inline Half operator-(const Half& a) {
  return_half(-a.y);
}

inline Half& operator+=(Half& a, const Half& b) {
  a.y += b.y;
  return a;
}

inline Half& operator-=(Half& a, const Half& b) {
  a.y -= b.y;
  return a;
}

inline Half& operator*=(Half& a, const Half& b) {
  a.y *= b.y;
  return a;
}

inline Half& operator/=(Half& a, const Half& b) {
  a.y /= b.y;
  return a;
}

#else

inline Half operator+(const Half& a, const Half& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline Half operator-(const Half& a, const Half& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline Half operator*(const Half& a, const Half& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline Half operator/(const Half& a, const Half& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline Half operator-(const Half& a) {
  return -static_cast<float>(a);
}

inline Half& operator+=(Half& a, const Half& b) {
  a = a + b;
  return a;
}

inline Half& operator-=(Half& a, const Half& b) {
  a = a - b;
  return a;
}

inline Half& operator*=(Half& a, const Half& b) {
  a = a * b;
  return a;
}

inline Half& operator/=(Half& a, const Half& b) {
  a = a / b;
  return a;
}

#endif

/// Arithmetic with floats

inline float operator+(Half a, float b) {
  return static_cast<float>(a) + b;
}
inline float operator-(Half a, float b) {
  return static_cast<float>(a) - b;
}
inline float operator*(Half a, float b) {
  return static_cast<float>(a) * b;
}
inline float operator/(Half a, float b) {
  return static_cast<float>(a) / b;
}

inline float operator+(float a, Half b) {
  return a + static_cast<float>(b);
}
inline float operator-(float a, Half b) {
  return a - static_cast<float>(b);
}
inline float operator*(float a, Half b) {
  return a * static_cast<float>(b);
}
inline float operator/(float a, Half b) {
  return a / static_cast<float>(b);
}

inline float& operator+=(float& a, const Half& b) {
  return a += static_cast<float>(b);
}
inline float& operator-=(float& a, const Half& b) {
  return a -= static_cast<float>(b);
}
inline float& operator*=(float& a, const Half& b) {
  return a *= static_cast<float>(b);
}
inline float& operator/=(float& a, const Half& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline double operator+(Half a, double b) {
  return static_cast<double>(a) + b;
}
inline double operator-(Half a, double b) {
  return static_cast<double>(a) - b;
}
inline double operator*(Half a, double b) {
  return static_cast<double>(a) * b;
}
inline double operator/(Half a, double b) {
  return static_cast<double>(a) / b;
}

inline double operator+(double a, Half b) {
  return a + static_cast<double>(b);
}
inline double operator-(double a, Half b) {
  return a - static_cast<double>(b);
}
inline double operator*(double a, Half b) {
  return a * static_cast<double>(b);
}
inline double operator/(double a, Half b) {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

#ifdef NATIVE_FP16

inline Half operator+(Half a, int32_t b) {
  return_half(a.y + b);
}
inline Half operator-(Half a, int32_t b) {
  return_half(a.y - b);
}
inline Half operator*(Half a, int32_t b) {
  return_half(a.y * b);
}
inline Half operator/(Half a, int32_t b) {
  return_half(a.y / b);
}

inline Half operator+(int32_t a, Half b) {
  return_half(a + b.y);
}
inline Half operator-(int32_t a, Half b) {
  return_half(a - b.y);
}
inline Half operator*(int32_t a, Half b) {
  return_half(a * b.y);
}
inline Half operator/(int32_t a, Half b) {
  return_half(a / b.y);
}

#else

inline Half operator+(Half a, int32_t b) {
  return a + static_cast<Half>(b);
}
inline Half operator-(Half a, int32_t b) {
  return a - static_cast<Half>(b);
}
inline Half operator*(Half a, int32_t b) {
  return a * static_cast<Half>(b);
}
inline Half operator/(Half a, int32_t b) {
  return a / static_cast<Half>(b);
}

inline Half operator+(int32_t a, Half b) {
  return static_cast<Half>(a) + b;
}
inline Half operator-(int32_t a, Half b) {
  return static_cast<Half>(a) - b;
}
inline Half operator*(int32_t a, Half b) {
  return static_cast<Half>(a) * b;
}
inline Half operator/(int32_t a, Half b) {
  return static_cast<Half>(a) / b;
}

#endif

//// Arithmetic with int64_t

#ifdef NATIVE_FP16

inline Half operator+(Half a, int64_t b) {
  return_half(a.y + b);
}
inline Half operator-(Half a, int64_t b) {
  return_half(a.y - b);
}
inline Half operator*(Half a, int64_t b) {
  return_half(a.y * b);
}
inline Half operator/(Half a, int64_t b) {
  return_half(a.y / b);
}

inline Half operator+(int64_t a, Half b) {
  return_half(a + b.y);
}
inline Half operator-(int64_t a, Half b) {
  return_half(a - b.y);
}
inline Half operator*(int64_t a, Half b) {
  return_half(a * b.y);
}
inline Half operator/(int64_t a, Half b) {
  return_half(a / b.y);
}

#else

inline Half operator+(Half a, int64_t b) {
  return a + static_cast<Half>(b);
}
inline Half operator-(Half a, int64_t b) {
  return a - static_cast<Half>(b);
}
inline Half operator*(Half a, int64_t b) {
  return a * static_cast<Half>(b);
}
inline Half operator/(Half a, int64_t b) {
  return a / static_cast<Half>(b);
}

inline Half operator+(int64_t a, Half b) {
  return static_cast<Half>(a) + b;
}
inline Half operator-(int64_t a, Half b) {
  return static_cast<Half>(a) - b;
}
inline Half operator*(int64_t a, Half b) {
  return static_cast<Half>(a) * b;
}
inline Half operator/(int64_t a, Half b) {
  return static_cast<Half>(a) / b;
}

#endif

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion Half to float.

static inline std::ostream& operator<<(
    std::ostream& out,
    const executorch::runtime::etensor::Half& value) {
  out << (float)value;
  return out;
}

} // namespace etensor
} // namespace runtime
} // namespace executorch
namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::Half;
} // namespace executor
} // namespace torch

namespace std {

template <>
class numeric_limits<executorch::runtime::etensor::Half> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;
  static constexpr executorch::runtime::etensor::Half min() {
    return executorch::runtime::etensor::Half(
        0x0400, executorch::runtime::etensor::Half::from_bits());
  }
  static constexpr executorch::runtime::etensor::Half lowest() {
    return executorch::runtime::etensor::Half(
        0xFBFF, executorch::runtime::etensor::Half::from_bits());
  }
  static constexpr executorch::runtime::etensor::Half max() {
    return executorch::runtime::etensor::Half(
        0x7BFF, executorch::runtime::etensor::Half::from_bits());
  }
  static constexpr executorch::runtime::etensor::Half epsilon() {
    return executorch::runtime::etensor::Half(
        0x1400, executorch::runtime::etensor::Half::from_bits());
  }
  static constexpr executorch::runtime::etensor::Half round_error() {
    return executorch::runtime::etensor::Half(
        0x3800, executorch::runtime::etensor::Half::from_bits());
  }
  static constexpr executorch::runtime::etensor::Half infinity() {
    return executorch::runtime::etensor::Half(
        0x7C00, executorch::runtime::etensor::Half::from_bits());
  }
  static constexpr executorch::runtime::etensor::Half quiet_NaN() {
    return executorch::runtime::etensor::Half(
        0x7E00, executorch::runtime::etensor::Half::from_bits());
  }
  static constexpr executorch::runtime::etensor::Half signaling_NaN() {
    return executorch::runtime::etensor::Half(
        0x7D00, executorch::runtime::etensor::Half::from_bits());
  }
  static constexpr executorch::runtime::etensor::Half denorm_min() {
    return executorch::runtime::etensor::Half(
        0x0001, executorch::runtime::etensor::Half::from_bits());
  }
};

} // namespace std
