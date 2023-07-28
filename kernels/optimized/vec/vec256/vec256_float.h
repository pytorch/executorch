/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <executorch/kernels/optimized/vec/intrinsics.h>
#include <executorch/kernels/optimized/vec/vec_base.h>

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace executorch {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <> class Vectorized<float> {
private:
  __m256 values;
public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }
  Vectorized() {}
  Vectorized(__m256 v) : values(v) {}
  Vectorized(float val) {
    values = _mm256_set1_ps(val);
  }
  Vectorized(float val1, float val2, float val3, float val4,
         float val5, float val6, float val7, float val8) {
    values = _mm256_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  operator __m256() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_blend_ps(a.values, b.values, mask);
  }
  static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask) {
    return _mm256_blendv_ps(a.values, b.values, mask.values);
  }
  template<typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<float>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
  }
  static Vectorized<float> set(const Vectorized<float>& a, const Vectorized<float>& b,
                           int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
    }
    return b;
  }
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
    __at_align__ float tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (size_t i = 0; i < size(); ++i) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
    return _mm256_loadu_ps(tmp_values);
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      float tmp_values[size()];
      _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
  const float& operator[](int idx) const  = delete;
  float& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    __m256 cmp = _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_EQ_OQ);
    return _mm256_movemask_ps(cmp);
  }
  Vectorized<float> isnan() const {
    return _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_UNORD_Q);
  }
  Vectorized<float> map(float (*const f)(float)) const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (size_t i = 0; i < size(); ++i) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> abs() const {
    auto mask = _mm256_set1_ps(-0.f);
    return _mm256_andnot_ps(mask, values);
  }
  Vectorized<float> acos() const {
    return Vectorized<float>(Sleef_acosf8_u10(values));
  }
  Vectorized<float> asin() const {
    return Vectorized<float>(Sleef_asinf8_u10(values));
  }
  Vectorized<float> atan() const {
    return Vectorized<float>(Sleef_atanf8_u10(values));
  }
  Vectorized<float> atan2(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_atan2f8_u10(values, b));
  }
  Vectorized<float> copysign(const Vectorized<float> &sign) const {
    return Vectorized<float>(Sleef_copysignf8(values, sign));
  }
  Vectorized<float> erf() const {
    // constants
    const auto neg_zero_vec = _mm256_set1_ps(-0.f);
    const auto one_vec = _mm256_set1_ps(1.0f);
    const auto p = _mm256_set1_ps(0.3275911f);
    const auto p1 = _mm256_set1_ps(0.254829592f);
    const auto p2 = _mm256_set1_ps(-0.284496736f);
    const auto p3 = _mm256_set1_ps(1.421413741f);
    const auto p4 = _mm256_set1_ps(-1.453152027f);
    const auto p5 = _mm256_set1_ps(1.061405429f);
    // sign(x)
    auto sign_mask = _mm256_and_ps(neg_zero_vec, values);
    auto abs_vec = _mm256_xor_ps(sign_mask, values);
    // t = 1 / (p * abs(x) + 1)
    auto tmp0 = _mm256_fmadd_ps(p, abs_vec, one_vec);
    auto t = _mm256_div_ps(one_vec, tmp0);
    // r = p5 * t ^ 4 + p4 * t ^ 3 + p3 * t ^ 2 + p2 * t + p1
    auto tmp1 = _mm256_fmadd_ps(p5, t, p4);
    auto tmp2 = _mm256_fmadd_ps(tmp1, t, p3);
    auto tmp3 = _mm256_fmadd_ps(tmp2, t, p2);
    auto r = _mm256_fmadd_ps(tmp3, t, p1);
    // - exp(- x * x)
    auto pow_2 = _mm256_mul_ps(values, values);
    auto neg_pow_2 = _mm256_xor_ps(neg_zero_vec, pow_2);
    // auto tmp4 = exp(neg_pow_2);
    auto tmp4 = Vectorized<float>(Sleef_expf8_u10(neg_pow_2));
    auto tmp5 = _mm256_xor_ps(neg_zero_vec, tmp4);
    // erf(x) = sign(x) * (1 - r * t * exp(- x * x))
    auto tmp6 = _mm256_mul_ps(tmp5, t);
    auto tmp7 = _mm256_fmadd_ps(tmp6, r, one_vec);
    return _mm256_xor_ps(sign_mask, tmp7);
  }
  Vectorized<float> erfc() const {
    return Vectorized<float>(Sleef_erfcf8_u15(values));
  }
  Vectorized<float> exp() const {
    return Vectorized<float>(Sleef_expf8_u10(values));
  }
  Vectorized<float> exp2() const {
    return Vectorized<float>(Sleef_exp2f8_u10(values));
  }
  Vectorized<float> expm1() const {
    return Vectorized<float>(Sleef_expm1f8_u10(values));
  }
  Vectorized<float> fmod(const Vectorized<float>& q) const {
    return Vectorized<float>(Sleef_fmodf8(values, q));
  }
  Vectorized<float> log() const {
    return Vectorized<float>(Sleef_logf8_u10(values));
  }
  Vectorized<float> log2() const {
    return Vectorized<float>(Sleef_log2f8_u10(values));
  }
  Vectorized<float> log10() const {
    return Vectorized<float>(Sleef_log10f8_u10(values));
  }
  Vectorized<float> log1p() const {
    return Vectorized<float>(Sleef_log1pf8_u10(values));
  }
  Vectorized<float> frac() const;
  Vectorized<float> sin() const {
    return Vectorized<float>(Sleef_sinf8_u35(values));
  }
  Vectorized<float> sinh() const {
    return Vectorized<float>(Sleef_sinhf8_u10(values));
  }
  Vectorized<float> cos() const {
    return Vectorized<float>(Sleef_cosf8_u35(values));
  }
  Vectorized<float> cosh() const {
    return Vectorized<float>(Sleef_coshf8_u10(values));
  }
  Vectorized<float> ceil() const {
    return _mm256_ceil_ps(values);
  }
  Vectorized<float> floor() const {
    return _mm256_floor_ps(values);
  }
  Vectorized<float> hypot(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_hypotf8_u05(values, b));
  }
  Vectorized<float> neg() const {
    return _mm256_xor_ps(_mm256_set1_ps(-0.f), values);
  }
  Vectorized<float> nextafter(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_nextafterf8(values, b));
  }
  Vectorized<float> round() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> tan() const {
    return Vectorized<float>(Sleef_tanf8_u10(values));
  }
  Vectorized<float> tanh() const {
    return Vectorized<float>(Sleef_tanhf8_u10(values));
  }
  Vectorized<float> trunc() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> lgamma() const {
    return Vectorized<float>(Sleef_lgammaf8_u10(values));
  }
  Vectorized<float> sqrt() const {
    return _mm256_sqrt_ps(values);
  }
  Vectorized<float> reciprocal() const {
    return _mm256_div_ps(_mm256_set1_ps(1), values);
  }
  Vectorized<float> rsqrt() const {
    return _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(values));
  }
  Vectorized<float> pow(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_powf8_u10(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<float> operator==(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ);
  }

  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_NEQ_UQ);
  }

  Vectorized<float> operator<(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LT_OQ);
  }

  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LE_OQ);
  }

  Vectorized<float> operator>(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GT_OQ);
  }

  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GE_OQ);
  }

  Vectorized<float> eq(const Vectorized<float>& other) const;
  Vectorized<float> ne(const Vectorized<float>& other) const;
  Vectorized<float> gt(const Vectorized<float>& other) const;
  Vectorized<float> ge(const Vectorized<float>& other) const;
  Vectorized<float> lt(const Vectorized<float>& other) const;
  Vectorized<float> le(const Vectorized<float>& other) const;
};

template <>
Vectorized<float> inline operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_add_ps(a, b);
}

template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_sub_ps(a, b);
}

template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_mul_ps(a, b);
}

template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_div_ps(a, b);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  Vectorized<float> max = _mm256_max_ps(a, b);
  Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  Vectorized<float> min = _mm256_min_ps(a, b);
  Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(min, isnan);
}

template <>
Vectorized<float> inline clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return _mm256_min_ps(max, _mm256_max_ps(min, a));
}

template <>
Vectorized<float> inline clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return _mm256_min_ps(max, a);
}

template <>
Vectorized<float> inline clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return _mm256_max_ps(min, a);
}

template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_and_ps(a, b);
}

template <>
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_or_ps(a, b);
}

template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_xor_ps(a, b);
}

inline Vectorized<float> Vectorized<float>::eq(const Vectorized<float>& other) const {
  return (*this == other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ne(const Vectorized<float>& other) const {
  return (*this != other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::gt(const Vectorized<float>& other) const {
  return (*this > other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ge(const Vectorized<float>& other) const {
  return (*this >= other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::lt(const Vectorized<float>& other) const {
  return (*this < other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::le(const Vectorized<float>& other) const {
  return (*this <= other) & Vectorized<float>(1.0f);
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}


template <>
Vectorized<float> inline fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return _mm256_fmadd_ps(a, b, c);
}

template <>
Vectorized<float> inline fmsub(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return _mm256_fmsub_ps(a, b, c);
}

// Used by Inductor CPP codegen
template<>
inline void transpose_mxn<float, 8, 8>(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  // load from src to registers
  // a: a0  a1  a2  a3  a4  a5  a6  a7
  // b: b0  b1  b2  b3  b4  b5  b6  b7
  // c: c0  c1  c2  c3  c4  c5  c6  c7
  // d: d0  d1  d2  d3  d4  d5  d6  d7
  // e: e0  e1  e2  e3  e4  e5  e6  e7
  // f: f0  f1  f2  f3  f4  f5  f6  f7
  // g: g0  g1  g2  g3  g4  g5  g6  g7
  // h: h0  h1  h2  h3  h4  h5  h6  h7
  __m256 a = _mm256_loadu_ps(&src[0 * ld_src]);
  __m256 b = _mm256_loadu_ps(&src[1 * ld_src]);
  __m256 c = _mm256_loadu_ps(&src[2 * ld_src]);
  __m256 d = _mm256_loadu_ps(&src[3 * ld_src]);
  __m256 e = _mm256_loadu_ps(&src[4 * ld_src]);
  __m256 f = _mm256_loadu_ps(&src[5 * ld_src]);
  __m256 g = _mm256_loadu_ps(&src[6 * ld_src]);
  __m256 h = _mm256_loadu_ps(&src[7 * ld_src]);

  __m256 ta, tb, tc, td, te, tf, tg, th;
  // unpacking and interleaving 32-bit elements
  // a0  b0  a1  b1  a4  b4  a5  b5
  // a2  b2  a3  b3  a6  b6  a7  b7
  // c0  d0  c1  d1 ...
  // c2  d2  c3  d3 ...
  // e0  f0  e1  f1 ...
  // e2  f2  e3  f3 ...
  // g0  h0  g1  h1 ...
  // g2  h2  g3  h3 ...
  ta = _mm256_unpacklo_ps(a, b);
  tb = _mm256_unpackhi_ps(a, b);
  tc = _mm256_unpacklo_ps(c, d);
  td = _mm256_unpackhi_ps(c, d);
  te = _mm256_unpacklo_ps(e, f);
  tf = _mm256_unpackhi_ps(e, f);
  tg = _mm256_unpacklo_ps(g, h);
  th = _mm256_unpackhi_ps(g, h);

  // unpacking and interleaving 64-bit elements
  //  a0  b0  c0  d0  a4  b4  c4  d4
  //  a1  b1  c1  d1 ...
  //  a2  b2  c2  d2 ...
  //  a3  b3  c3  d3 ...
  //  e0  f0  g0  h0  e4  f4  g4  h4
  //  e1  f1  g1  h1 ...
  //  e2  f2  g2  h2 ...
  //  e3  f3  g3  h3 ...
  a = _mm256_castpd_ps(
      _mm256_unpacklo_pd(_mm256_castps_pd(ta), _mm256_castps_pd(tc)));
  b = _mm256_castpd_ps(
      _mm256_unpackhi_pd(_mm256_castps_pd(ta), _mm256_castps_pd(tc)));
  c = _mm256_castpd_ps(
      _mm256_unpacklo_pd(_mm256_castps_pd(tb), _mm256_castps_pd(td)));
  d = _mm256_castpd_ps(
      _mm256_unpackhi_pd(_mm256_castps_pd(tb), _mm256_castps_pd(td)));
  e = _mm256_castpd_ps(
      _mm256_unpacklo_pd(_mm256_castps_pd(te), _mm256_castps_pd(tg)));
  f = _mm256_castpd_ps(
      _mm256_unpackhi_pd(_mm256_castps_pd(te), _mm256_castps_pd(tg)));
  g = _mm256_castpd_ps(
      _mm256_unpacklo_pd(_mm256_castps_pd(tf), _mm256_castps_pd(th)));
  h = _mm256_castpd_ps(
      _mm256_unpackhi_pd(_mm256_castps_pd(tf), _mm256_castps_pd(th)));

  //  shuffle 128-bits (composed of 4 32-bit elements)
  //  a0  b0  c0  d0  e0  f0  g0  h0
  //  a1  b1  c1  d1 ...
  //  a2  b2  c2  d2 ...
  //  a3  b3  c3  d3 ...
  //  a4  b4  c4  d4 ...
  //  a5  b5  c5  d5 ...
  //  a6  b6  c6  d6 ...
  //  a7  b7  c7  d7 ...
  ta = _mm256_permute2f128_ps(a, e, 0x20);
  tb = _mm256_permute2f128_ps(b, f, 0x20);
  tc = _mm256_permute2f128_ps(c, g, 0x20);
  td = _mm256_permute2f128_ps(d, h, 0x20);
  te = _mm256_permute2f128_ps(a, e, 0x31);
  tf = _mm256_permute2f128_ps(b, f, 0x31);
  tg = _mm256_permute2f128_ps(c, g, 0x31);
  th = _mm256_permute2f128_ps(d, h, 0x31);

  // store from registers to dst
  _mm256_storeu_ps(&dst[0 * ld_dst], ta);
  _mm256_storeu_ps(&dst[1 * ld_dst], tb);
  _mm256_storeu_ps(&dst[2 * ld_dst], tc);
  _mm256_storeu_ps(&dst[3 * ld_dst], td);
  _mm256_storeu_ps(&dst[4 * ld_dst], te);
  _mm256_storeu_ps(&dst[5 * ld_dst], tf);
  _mm256_storeu_ps(&dst[6 * ld_dst], tg);
  _mm256_storeu_ps(&dst[7 * ld_dst], th);
}

#endif

}}}
