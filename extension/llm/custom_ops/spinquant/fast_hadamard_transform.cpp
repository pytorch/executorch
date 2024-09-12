/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fast_hadamard_transform.h"

#include <algorithm>

namespace executorch {
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

  // Per the function-level comment in the header, we can ignore the
  // quantization scale, so we just delegate to the usual unnormalized
  // implementation.
  // NOTE: if we need this to be fast on CPU, we can use FFHT to
  // generate fht_uint32 similar to fht_float.
  internal::fast_hadamard_transform_unnormalized_simple_impl(
      tmp.get(), log2_vec_size);

  // Normalization step: divide by sqrt(1 << log2_vec_size). Similar
  // to fast_sqrt, if N is even, then the maximum-precision way
  // to do this is right-shift by log2_vec_size / 2. If N is odd, we
  // still do the right-shift, and then we have an extra division by
  // sqrt(2) that we perform by making use of a sufficiently accurate
  // rational approximation. (Our initial idea was to divide by sqrt(2)
  // by adjusting the quantization scale, but that would cause this
  // function to tend to increase the magnitude of the elements of
  // vec, which would resulting in clipping and therefore accuracy
  // loss, especially compounded over 30+ transformer layers.)
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
      vec[ii] = std::clamp(val_over_sqrt_vec_size, qmin, qmax);
    }
  } else {
    for (int ii = 0; ii < vec_size; ++ii) {
      vec[ii] = std::clamp(tmp[ii] >> log2_sqrt_vec_size, qmin, qmax);
    }
  }
  return;
}
} // namespace executorch
