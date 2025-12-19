/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <executorch/backends/aoti/slim/c10/util/BFloat16.h>
#include <executorch/backends/aoti/slim/c10/util/Float8_e4m3fn.h>
#include <executorch/backends/aoti/slim/c10/util/Float8_e4m3fnuz.h>
#include <executorch/backends/aoti/slim/c10/util/Float8_e5m2.h>
#include <executorch/backends/aoti/slim/c10/util/Float8_e5m2fnuz.h>
#include <executorch/backends/aoti/slim/c10/util/Float8_e8m0fnu.h>
#include <executorch/backends/aoti/slim/c10/util/Half.h>
#include <executorch/backends/aoti/slim/c10/util/complex.h>
#include <executorch/backends/aoti/slim/c10/util/overflows.h>
#include <executorch/runtime/platform/assert.h>

#include <type_traits>

STANDALONE_CLANG_DIAGNOSTIC_PUSH()
#if STANDALONE_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
STANDALONE_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif
#if STANDALONE_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
STANDALONE_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace executorch::backends::aoti::slim::c10 {

template <typename dest_t, typename src_t>
struct needs_real {
  constexpr static bool value =
      (is_complex<src_t>::value && !is_complex<dest_t>::value);
};

template <bool, typename src_t>
struct maybe_real {
  STANDALONE_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

template <typename src_t>
struct maybe_real<true, src_t> {
  STANDALONE_HOST_DEVICE static inline decltype(auto) apply(src_t src) {
    return src.real();
  }
};

template <bool, typename src_t>
struct maybe_bool {
  STANDALONE_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

template <typename src_t>
struct maybe_bool<true, src_t> {
  STANDALONE_HOST_DEVICE static inline decltype(auto) apply(src_t src) {
    // Don't use bool operator so as to to also compile for ComplexHalf.
    return src.real() || src.imag();
  }
};

// Note: deliberately ignores undefined behavior, consistent with NumPy.
// PyTorch's type conversions can cause a variety of undefined behavior,
// including float to integral overflow and signed to unsigned integer overflow.
// Some of this undefined behavior is addressed below.
template <typename dest_t, typename src_t>
struct static_cast_with_inter_type {
  STANDALONE_HOST_DEVICE __ubsan_ignore_undefined__ static inline dest_t apply(
      src_t src) {
    constexpr bool real = needs_real<dest_t, src_t>::value;
    auto r = maybe_real<real, src_t>::apply(src);
    return static_cast<dest_t>(r);
  }
};

// Partial template specialization for casting to bool.
// Need to handle complex types separately, as we don't
// simply want to cast the real part to bool.
template <typename src_t>
struct static_cast_with_inter_type<bool, src_t> {
  STANDALONE_HOST_DEVICE static inline bool apply(src_t src) {
    constexpr bool complex = needs_real<bool, src_t>::value;
    return static_cast<bool>(maybe_bool<complex, src_t>::apply(src));
  }
};

// Partial template instantiation for casting to uint8.
// Note: Converting from negative float values to unsigned integer types is
// undefined behavior in C++, and current CPU and GPU compilers exhibit
// divergent behavior. Casting from negative float values to signed
// integer types and then to unsigned integer types is not undefined,
// however, so this cast improves the consistency of type conversions
// to uint8 across compilers.
// Further note: Type conversions across compilers still have other undefined
// and divergent behavior.
template <typename src_t>
struct static_cast_with_inter_type<uint8_t, src_t> {
  STANDALONE_HOST_DEVICE __ubsan_ignore_undefined__ static inline uint8_t apply(
      src_t src) {
    constexpr bool real = needs_real<uint8_t, src_t>::value;
    return static_cast<uint8_t>(
        static_cast<int64_t>(maybe_real<real, src_t>::apply(src)));
  }
};

template <>
struct static_cast_with_inter_type<
    executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>,
    executorch::backends::aoti::slim::c10::BFloat16> {
  STANDALONE_HOST_DEVICE
  __ubsan_ignore_undefined__ static inline executorch::backends::aoti::slim::
      c10::complex<executorch::backends::aoti::slim::c10::Half>
      apply(executorch::backends::aoti::slim::c10::BFloat16 src) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>>(
        executorch::backends::aoti::slim::c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>,
    executorch::backends::aoti::slim::c10::Float8_e5m2> {
  STANDALONE_HOST_DEVICE
  __ubsan_ignore_undefined__ static inline executorch::backends::aoti::slim::
      c10::complex<executorch::backends::aoti::slim::c10::Half>
      apply(executorch::backends::aoti::slim::c10::Float8_e5m2 src) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>>(
        executorch::backends::aoti::slim::c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>,
    executorch::backends::aoti::slim::c10::Float8_e5m2fnuz> {
  STANDALONE_HOST_DEVICE
  __ubsan_ignore_undefined__ static inline executorch::backends::aoti::slim::
      c10::complex<executorch::backends::aoti::slim::c10::Half>
      apply(executorch::backends::aoti::slim::c10::Float8_e5m2fnuz src) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>>(
        executorch::backends::aoti::slim::c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>,
    executorch::backends::aoti::slim::c10::Float8_e4m3fn> {
  STANDALONE_HOST_DEVICE
  __ubsan_ignore_undefined__ static inline executorch::backends::aoti::slim::
      c10::complex<executorch::backends::aoti::slim::c10::Half>
      apply(executorch::backends::aoti::slim::c10::Float8_e4m3fn src) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>>(
        executorch::backends::aoti::slim::c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>,
    executorch::backends::aoti::slim::c10::Float8_e4m3fnuz> {
  STANDALONE_HOST_DEVICE
  __ubsan_ignore_undefined__ static inline executorch::backends::aoti::slim::
      c10::complex<executorch::backends::aoti::slim::c10::Half>
      apply(executorch::backends::aoti::slim::c10::Float8_e4m3fnuz src) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>>(
        executorch::backends::aoti::slim::c10::complex<float>{src});
  }
};

// TODO(#146647): Can we make all these template specialization happen
// based off our apply macros?
template <>
struct static_cast_with_inter_type<
    executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>,
    executorch::backends::aoti::slim::c10::Float8_e8m0fnu> {
  STANDALONE_HOST_DEVICE
  __ubsan_ignore_undefined__ static inline executorch::backends::aoti::slim::
      c10::complex<executorch::backends::aoti::slim::c10::Half>
      apply(executorch::backends::aoti::slim::c10::Float8_e8m0fnu src) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>>(
        executorch::backends::aoti::slim::c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>,
    executorch::backends::aoti::slim::c10::Half> {
  STANDALONE_HOST_DEVICE
  __ubsan_ignore_undefined__ static inline executorch::backends::aoti::slim::
      c10::complex<executorch::backends::aoti::slim::c10::Half>
      apply(executorch::backends::aoti::slim::c10::Half src) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>>(
        executorch::backends::aoti::slim::c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>,
    executorch::backends::aoti::slim::c10::complex<double>> {
  STANDALONE_HOST_DEVICE
  __ubsan_ignore_undefined__ static inline executorch::backends::aoti::slim::
      c10::complex<executorch::backends::aoti::slim::c10::Half>
      apply(executorch::backends::aoti::slim::c10::complex<double> src) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<
        executorch::backends::aoti::slim::c10::Half>>(
        static_cast<executorch::backends::aoti::slim::c10::complex<float>>(
            src));
  }
};

template <typename To, typename From>
STANDALONE_HOST_DEVICE To convert(From f) {
  return static_cast_with_inter_type<To, From>::apply(f);
}

// Define separately to avoid being inlined and prevent code-size bloat
[[noreturn]] inline void report_overflow(const char* name) {
  ET_CHECK_MSG(
      false, "value cannot be converted to type %s without overflow", name);
}

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  // Converting to bool can't overflow so we exclude this case from checking.
  if (!std::is_same_v<To, bool> &&
      overflows<To, From>(f, /* strict_unsigned */ !std::is_signed_v<To>)) {
    report_overflow(name);
  }
  return convert<To, From>(f);
}

} // namespace executorch::backends::aoti::slim::c10

STANDALONE_CLANG_DIAGNOSTIC_POP()

// Trigger tests for D25440771. TODO: Remove this line any time you want.
