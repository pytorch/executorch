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

#include <executorch/extension/llm/custom_ops/spinquant/third-party/FFHT/fht.h>

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

template <typename T>
void fast_hadamard_transform_unnormalized_simple_impl(
    T* vec,
    int log2_vec_size) {
  // NOTE: If you're here because you're profiling a model and this is
  // slow, consider updating FFHT to generate efficient assembly for
  // your data type!
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

inline void fast_hadamard_transform_ffht_impl(float* vec, int log2_vec_size) {
#if defined(__aarch64__) || defined(__x86_64__)
  if (log2_vec_size <= 0) {
    return;
  }

  fht_float(vec, log2_vec_size);
  normalize_after_fht(vec, log2_vec_size);
#else
  fast_hadamard_transform_simple_impl(vec, log2_vec_size);
#endif
}

} // namespace internal

// Compute the fast Walsh-Hadamard transform
// (https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform)
// of vec, which must be of length (1 << log2_vec_size).
template <typename T>
void fast_hadamard_transform(T* vec, int log2_vec_size) {
  if constexpr (std::is_same_v<T, float>) {
    internal::fast_hadamard_transform_ffht_impl(vec, log2_vec_size);
  } else {
    internal::fast_hadamard_transform_simple_impl(vec, log2_vec_size);
  }
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
    int log2_vec_size);

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
    int log2_vec_size);

} // namespace executorch
