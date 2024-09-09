/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>

#include "fast_hadamard_transform_special.h"

namespace executorch {
namespace internal {

// Square root of 1 << log2_n.
template <typename T>
T fast_sqrt_of_power_of_2(int log2_n) {
  // The square root of 2**N is, by definition, 2**(N/2), which is
  // trivial to compute for even N using a left shift.
  //
  // For odd N, 2**(N/2) = 2**(floor(N/2) + 1/2)
  //                     = 2**(floor(N/2)) * (2 ** (1/2))
  //                     = 2**(floor(N/2)) * sqrt(2)
  // which is again fast to compute.
  return T(1 << (log2_n / 2)) * ((log2_n % 2) ? T(std::sqrt(2)) : T(1));
}

template <typename T>
void normalize_after_fht(T* out, int log2_vec_size) {
  const T inv_sqrt = T(1) / fast_sqrt_of_power_of_2<T>(log2_vec_size);
  const int vec_size = 1 << log2_vec_size;
  for (int ii = 0; ii < vec_size; ++ii) {
    out[ii] *= inv_sqrt;
  }
}

// Normalization step: divide by sqrt(1 << log2_vec_size). Similar
// to fast_sqrt above, if N is even, then the maximum-precision way
// to do this is right-shift by log2_vec_size / 2. If N is odd, we
// still do the right-shift, and then we have an extra division by
// sqrt(2) that we perform by making use of a sufficiently accurate
// rational approximation. Our initial idea was to divide by sqrt(2)
// by adjusting the quantization scale, but that would cause this
// function to tend to increase the magnitude of the elements of
// vec, which would resulting in clipping and therefore accuracy
// loss, especially compounded over 30+ transformer layers.
void quantized_normalize_after_fht(
    const int32_t* tmp,
    int16_t* out,
    int log2_vec_size,
    int vec_size) {
  const int log2_sqrt_vec_size = log2_vec_size / 2;
  constexpr int32_t qmin = -(1 << 15) + 1;
  constexpr int32_t qmax = -qmin;
  if (log2_vec_size % 2 != 0) {
    // 408 / 577 - 1.0 / sqrt(2) ~= 1.062e-0.6, which should be close enough.
    static const int32_t inv_sqrt_2_numerator = 408;
    static const int32_t inv_sqrt_2_denominator = 577;
    for (int ii = 0; ii < vec_size; ++ii) {
      const auto val_over_sqrt_vec_size =
          (tmp[ii] * inv_sqrt_2_numerator / inv_sqrt_2_denominator) >>
          log2_sqrt_vec_size;
      out[ii] = std::clamp(val_over_sqrt_vec_size, qmin, qmax);
    }
  } else {
    for (int ii = 0; ii < vec_size; ++ii) {
      out[ii] = std::clamp(tmp[ii] >> log2_sqrt_vec_size, qmin, qmax);
    }
  }
}

template <typename T>
void fast_hadamard_transform_unnormalized_simple_impl(
    T* vec,
    int log2_vec_size) {
  if (log2_vec_size == 0) {
    return;
  }

  int step = 1;
  const auto vec_size = 1 << log2_vec_size;
  while (step < vec_size) {
    for (int ii = 0; ii < vec_size; ii += step * 2) {
      for (int jj = ii; jj < ii + step; ++jj) {
        auto x = vec[jj];
        auto y = vec[jj + step];
        vec[jj] = x + y;
        vec[jj + step] = x - y;
      }
    }
    step *= 2;
  }
}

template <typename T>
void fast_hadamard_transform_simple_impl(T* vec, int log2_vec_size) {
  fast_hadamard_transform_unnormalized_simple_impl(vec, log2_vec_size);
  normalize_after_fht(vec, log2_vec_size);
}

} // namespace internal

// Compute the fast Walsh-Hadamard transform
// (https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform)
// of vec, which must be of length (1 << log2_vec_size).
template <typename T>
void fast_hadamard_transform(T* vec, int log2_vec_size) {
  internal::fast_hadamard_transform_simple_impl(vec, log2_vec_size);
}

// Compute a quantized fast Walsh-Hadamard transform of vec, which
// must be of length (1 << log2_vec_size) and symmetrically quantized.
//
// Note that we do not need to know the quantization scale, because
// the Fast Hadamard transform is a series of additions and
// subtractions with a final multiplication step, and we have the
// following trivial identities:
//
// scale * a + scale * b = scale * (a + b)  (addition doesn't need the scale)
// alpha * (scale * a) = scale * (alpha * a) (multiplication doesn't need the
// scale)
void fast_hadamard_transform_symmetric_quantized_s16(
    int16_t* vec,
    int log2_vec_size) {
  if (log2_vec_size == 0) {
    return;
  }

  const int vec_size = 1 << log2_vec_size;
  // We perform log2_vec_size rounds where each round's maximum output
  // is at most double the maximum input, so we can at most multiply
  // the maximum input by vec_size. Performing intermediate arithmetic
  // in 32-bit precision should prevent overflow, since 16 +
  // log2_vec_size should be much less than 32.
  auto tmp = std::make_unique<int32_t[]>(vec_size);
  std::copy(vec, vec + vec_size, tmp.get());

  // Per the function-level comment above, we can ignore the
  // quantization scale, so we just delegate to the usual unnormalized
  // implementation.
  // NOTE: if we need this to be fast on CPU, we can use FFHT to
  // generate fht_uint32 similar to fht_float.
  internal::fast_hadamard_transform_unnormalized_simple_impl(
      tmp.get(), log2_vec_size);

  internal::quantized_normalize_after_fht(
      tmp.get(), vec, log2_vec_size, vec_size);
}

// Like fast_hadamard_transform, but vec must be of length 28 * (1 <<
// log2_vec_size) and the transform is computed by interpreting vec as
// a (28, 1 << log2_vec_size) matrix and performing 28 FHTs, followed
// by (1 << log2_vec_size) multiplications by a particular Hadamard
// matrix of size 28x28 (see special_hadamard_code_gen.py for the
// exact matrix).
template <typename T>
void fast_hadamard_transform_28N(T* vec, int log2_vec_size) {
  const int vec_size = (1 << log2_vec_size);
  for (int ii = 0; ii < 28; ++ii) {
    fast_hadamard_transform(&vec[ii * vec_size], log2_vec_size);
  }
  for (int ii = 0; ii < vec_size; ++ii) {
    hadamard_mult_28_strided(&vec[ii], vec_size);
  }
}

// We don't need the quantization scale; see the function-level
// comment on fast_hadamard_transform_symmetric_quantized_s16 for
// details.
void fast_hadamard_transform_symmetric_quantized_s16_28N(
    int16_t* vec,
    int log2_vec_size) {
  if (log2_vec_size == 0) {
    return;
  }
  const int vec_size = (1 << log2_vec_size);

  auto tmp = std::make_unique<int32_t[]>(vec_size);
  std::copy(vec, vec + vec_size * 28, tmp.get());

  for (int ii = 0; ii < 28; ++ii) {
    internal::fast_hadamard_transform_unnormalized_simple_impl(
        &tmp[ii * vec_size], log2_vec_size);
  }

  for (int ii = 0; ii < vec_size; ++ii) {
    hadamard_mult_28_strided(&tmp[ii], vec_size);
  }

  internal::quantized_normalize_after_fht(
      tmp.get(), vec, log2_vec_size, vec_size * 28);
}

} // namespace executorch
