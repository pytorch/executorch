/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if !defined(STANDALONE_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "standalone/c10/util/complex_math.h is not meant to be individually included. Include standalone/c10/util/complex.h instead."
#endif

#include <cmath>

namespace executorch::backends::aoti::slim::c10::complex_math {

// Exponential functions

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
exp(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::exp(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::exp(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
log(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::log(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::log(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
log10(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::log10(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::log10(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
log2(const executorch::backends::aoti::slim::c10::complex<T>& x) {
  const executorch::backends::aoti::slim::c10::complex<T> log2 =
      executorch::backends::aoti::slim::c10::complex<T>(::log(2.0), 0.0);
  return executorch::backends::aoti::slim::c10::complex_math::log(x) / log2;
}

// Power functions
//
#if defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX))
namespace _detail {
template <typename T>
executorch::backends::aoti::slim::c10::complex<T> compute_csqrt(
    const executorch::backends::aoti::slim::c10::complex<T>& z) {
  constexpr auto half = T(.5);

  // Trust standard library to correctly handle infs and NaNs
  if (std::isinf(z.real()) || std::isinf(z.imag()) || std::isnan(z.real()) ||
      std::isnan(z.imag())) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
        std::sqrt(static_cast<std::complex<T>>(z)));
  }

  // Special case for square root of pure imaginary values
  if (z.real() == T(0)) {
    if (z.imag() == T(0)) {
      return executorch::backends::aoti::slim::c10::complex<T>(T(0), z.imag());
    }
    auto v = std::sqrt(half * std::abs(z.imag()));
    return executorch::backends::aoti::slim::c10::complex<T>(
        v, std::copysign(v, z.imag()));
  }

  // At this point, z is non-zero and finite
  if (z.real() >= 0.0) {
    auto t = std::sqrt((z.real() + std::abs(z)) * half);
    return executorch::backends::aoti::slim::c10::complex<T>(
        t, half * (z.imag() / t));
  }

  auto t = std::sqrt((-z.real() + std::abs(z)) * half);
  return executorch::backends::aoti::slim::c10::complex<T>(
      half * std::abs(z.imag() / t), std::copysign(t, z.imag()));
}

// Compute complex arccosine using formula from W. Kahan
// "Branch Cuts for Complex Elementary Functions" 1986 paper:
// cacos(z).re = 2*atan2(sqrt(1-z).re(), sqrt(1+z).re())
// cacos(z).im = asinh((sqrt(conj(1+z))*sqrt(1-z)).im())
template <typename T>
executorch::backends::aoti::slim::c10::complex<T> compute_cacos(
    const executorch::backends::aoti::slim::c10::complex<T>& z) {
  auto constexpr one = T(1);
  // Trust standard library to correctly handle infs and NaNs
  if (std::isinf(z.real()) || std::isinf(z.imag()) || std::isnan(z.real()) ||
      std::isnan(z.imag())) {
    return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
        std::acos(static_cast<std::complex<T>>(z)));
  }
  auto a = compute_csqrt(executorch::backends::aoti::slim::c10::complex<T>(
      one - z.real(), -z.imag()));
  auto b = compute_csqrt(executorch::backends::aoti::slim::c10::complex<T>(
      one + z.real(), z.imag()));
  auto c = compute_csqrt(executorch::backends::aoti::slim::c10::complex<T>(
      one + z.real(), -z.imag()));
  auto r = T(2) * std::atan2(a.real(), b.real());
  // Explicitly unroll (a*c).imag()
  auto i = std::asinh(a.real() * c.imag() + a.imag() * c.real());
  return executorch::backends::aoti::slim::c10::complex<T>(r, i);
}

inline executorch::backends::aoti::slim::c10::complex<float> sqrt(
    const executorch::backends::aoti::slim::c10::complex<float>& in) {
  return compute_csqrt(in);
}

inline executorch::backends::aoti::slim::c10::complex<double> sqrt(
    const executorch::backends::aoti::slim::c10::complex<double>& in) {
  return compute_csqrt(in);
}

inline executorch::backends::aoti::slim::c10::complex<float> acos(
    const executorch::backends::aoti::slim::c10::complex<float>& in) {
  return compute_cacos(in);
}

inline executorch::backends::aoti::slim::c10::complex<double> acos(
    const executorch::backends::aoti::slim::c10::complex<double>& in) {
  return compute_cacos(in);
}
} // namespace _detail
#endif

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
sqrt(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::sqrt(static_cast<thrust::complex<T>>(x)));
#elif !(                        \
    defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX)))
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::sqrt(static_cast<std::complex<T>>(x)));
#else
  return _detail::sqrt(x);
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
pow(const executorch::backends::aoti::slim::c10::complex<T>& x,
    const executorch::backends::aoti::slim::c10::complex<T>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::pow(
          static_cast<thrust::complex<T>>(x),
          static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::pow(
          static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
pow(const executorch::backends::aoti::slim::c10::complex<T>& x, const T& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
pow(const T& x, const executorch::backends::aoti::slim::c10::complex<T>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::pow(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

template <typename T, typename U>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<
    decltype(T() * U())>
pow(const executorch::backends::aoti::slim::c10::complex<T>& x,
    const executorch::backends::aoti::slim::c10::complex<U>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::pow(
          static_cast<thrust::complex<T>>(x),
          static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::pow(
          static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template <typename T, typename U>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<
    decltype(T() * U())>
pow(const executorch::backends::aoti::slim::c10::complex<T>& x, const U& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template <typename T, typename U>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<
    decltype(T() * U())>
pow(const T& x, const executorch::backends::aoti::slim::c10::complex<U>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::pow(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

// Trigonometric functions

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
sin(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::sin(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::sin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
cos(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::cos(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::cos(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
tan(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::tan(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::tan(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
asin(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::asin(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::asin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
acos(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::acos(static_cast<thrust::complex<T>>(x)));
#elif !defined(_LIBCPP_VERSION)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::acos(static_cast<std::complex<T>>(x)));
#else
  return _detail::acos(x);
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
atan(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::atan(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::atan(static_cast<std::complex<T>>(x)));
#endif
}

// Hyperbolic functions

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
sinh(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::sinh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::sinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
cosh(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::cosh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::cosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
tanh(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::tanh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::tanh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
asinh(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::asinh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::asinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
acosh(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::acosh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::acosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
atanh(const executorch::backends::aoti::slim::c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      thrust::atanh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<executorch::backends::aoti::slim::c10::complex<T>>(
      std::atanh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
log1p(const executorch::backends::aoti::slim::c10::complex<T>& z) {
#if defined(__APPLE__) || defined(__MACOSX) || defined(__CUDACC__) || \
    defined(__HIPCC__)
  // For Mac, the new implementation yielded a high relative error. Falling back
  // to the old version for now.
  // See https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
  // For CUDA we also use this one, as thrust::log(thrust::complex) takes
  // *forever* to compile

  // log1p(z) = log(1 + z)
  // Let's define 1 + z = r * e ^ (i * a), then we have
  // log(r * e ^ (i * a)) = log(r) + i * a
  // With z = x + iy, the term r can be written as
  // r = ((1 + x) ^ 2 + y ^ 2) ^ 0.5
  //   = (1 + x ^ 2 + 2 * x + y ^ 2) ^ 0.5
  // So, log(r) is
  // log(r) = 0.5 * log(1 + x ^ 2 + 2 * x + y ^ 2)
  //        = 0.5 * log1p(x * (x + 2) + y ^ 2)
  // we need to use the expression only on certain condition to avoid overflow
  // and underflow from `(x * (x + 2) + y ^ 2)`
  T x = z.real();
  T y = z.imag();
  T zabs = std::abs(z);
  T theta = std::atan2(y, x + T(1));
  if (zabs < 0.5) {
    T r = x * (T(2) + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {T(0.5) * std::log1p(r), theta};
  } else {
    T z0 = std::hypot(x + 1, y);
    return {std::log(z0), theta};
  }
#else
  // CPU path
  // Based on https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
  executorch::backends::aoti::slim::c10::complex<T> u = z + T(1);
  if (u == T(1)) {
    return z;
  } else {
    auto log_u = log(u);
    if (u - T(1) == z) {
      return log_u;
    }
    return log_u * (z / (u - T(1)));
  }
#endif
}

template <typename T>
STANDALONE_HOST_DEVICE inline executorch::backends::aoti::slim::c10::complex<T>
expm1(const executorch::backends::aoti::slim::c10::complex<T>& z) {
  // expm1(z) = exp(z) - 1
  // Define z = x + i * y
  // f = e ^ (x + i * y) - 1
  //   = e ^ x * e ^ (i * y) - 1
  //   = (e ^ x * cos(y) - 1) + i * (e ^ x * sin(y))
  //   = (e ^ x - 1) * cos(y) - (1 - cos(y)) + i * e ^ x * sin(y)
  //   = expm1(x) * cos(y) - 2 * sin(y / 2) ^ 2 + i * e ^ x * sin(y)
  T x = z.real();
  T y = z.imag();
  T a = std::sin(y / 2);
  T er = std::expm1(x) * std::cos(y) - T(2) * a * a;
  T ei = std::exp(x) * std::sin(y);
  return {er, ei};
}

} // namespace executorch::backends::aoti::slim::c10::complex_math

using executorch::backends::aoti::slim::c10::complex_math::acos;
using executorch::backends::aoti::slim::c10::complex_math::acosh;
using executorch::backends::aoti::slim::c10::complex_math::asin;
using executorch::backends::aoti::slim::c10::complex_math::asinh;
using executorch::backends::aoti::slim::c10::complex_math::atan;
using executorch::backends::aoti::slim::c10::complex_math::atanh;
using executorch::backends::aoti::slim::c10::complex_math::cos;
using executorch::backends::aoti::slim::c10::complex_math::cosh;
using executorch::backends::aoti::slim::c10::complex_math::exp;
using executorch::backends::aoti::slim::c10::complex_math::expm1;
using executorch::backends::aoti::slim::c10::complex_math::log;
using executorch::backends::aoti::slim::c10::complex_math::log10;
using executorch::backends::aoti::slim::c10::complex_math::log1p;
using executorch::backends::aoti::slim::c10::complex_math::log2;
using executorch::backends::aoti::slim::c10::complex_math::pow;
using executorch::backends::aoti::slim::c10::complex_math::sin;
using executorch::backends::aoti::slim::c10::complex_math::sinh;
using executorch::backends::aoti::slim::c10::complex_math::sqrt;
using executorch::backends::aoti::slim::c10::complex_math::tan;
using executorch::backends::aoti::slim::c10::complex_math::tanh;

namespace std {

using executorch::backends::aoti::slim::c10::complex_math::acos;
using executorch::backends::aoti::slim::c10::complex_math::acosh;
using executorch::backends::aoti::slim::c10::complex_math::asin;
using executorch::backends::aoti::slim::c10::complex_math::asinh;
using executorch::backends::aoti::slim::c10::complex_math::atan;
using executorch::backends::aoti::slim::c10::complex_math::atanh;
using executorch::backends::aoti::slim::c10::complex_math::cos;
using executorch::backends::aoti::slim::c10::complex_math::cosh;
using executorch::backends::aoti::slim::c10::complex_math::exp;
using executorch::backends::aoti::slim::c10::complex_math::expm1;
using executorch::backends::aoti::slim::c10::complex_math::log;
using executorch::backends::aoti::slim::c10::complex_math::log10;
using executorch::backends::aoti::slim::c10::complex_math::log1p;
using executorch::backends::aoti::slim::c10::complex_math::log2;
using executorch::backends::aoti::slim::c10::complex_math::pow;
using executorch::backends::aoti::slim::c10::complex_math::sin;
using executorch::backends::aoti::slim::c10::complex_math::sinh;
using executorch::backends::aoti::slim::c10::complex_math::sqrt;
using executorch::backends::aoti::slim::c10::complex_math::tan;
using executorch::backends::aoti::slim::c10::complex_math::tanh;

} // namespace std
