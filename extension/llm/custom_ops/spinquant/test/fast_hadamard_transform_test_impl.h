/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <executorch/extension/llm/custom_ops/spinquant/fast_hadamard_transform.h>
#include <executorch/extension/llm/custom_ops/spinquant/test/fast_hadamard_transform_special_unstrided_cpu.h>

namespace executorch::runtime::testing {
void reference_fht_impl(float* buf, int n);

// Alternate implementation of fast_hadamard_transform_28N to mutation
// test against. Benchmarking suggests this one is slower, which is
// why it's in the test.
template <typename T>
void fast_hadamard_transform_28N_with_transpose(T* vec, int log2_vec_size) {
  const int vec_size = (1 << log2_vec_size);
  for (int ii = 0; ii < 28; ++ii) {
    executorch::fast_hadamard_transform(&vec[ii * vec_size], log2_vec_size);
  }
  std::unique_ptr<T[]> transposed = std::make_unique<T[]>(28 * vec_size);
  for (int ii = 0; ii < 28; ++ii) {
    for (int jj = 0; jj < vec_size; ++jj) {
      transposed[jj * 28 + ii] = vec[ii * vec_size + jj];
    }
  }
  for (int ii = 0; ii < vec_size; ++ii) {
    hadamard_mult_28(&transposed[ii * 28]);
  }
  for (int jj = 0; jj < vec_size; ++jj) {
    for (int ii = 0; ii < 28; ++ii) {
      vec[ii * vec_size + jj] = transposed[jj * 28 + ii];
    }
  }
}

std::vector<float> random_floats(int howMany);

} // namespace executorch::runtime::testing
