/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Slightly modified version of caffe2/aten/src/ATen/native/cpu/moments_utils.h
// for use in optimized ExecuTorch ops. Template specializations of BFloat16
// are excluded.

#include <executorch/kernels/optimized/vec/vec.h>

#include <executorch/kernels/optimized/utils/math_utils.h>
#include <executorch/runtime/platform/compiler.h>
#include <array>

namespace torch {
namespace executor {
namespace native {

template <typename T>
using acc_t = executorch::utils::compute_dtype<T>;

constexpr int64_t kChunkSize = 16;

template <typename T>
void AddMoments(
    int64_t m0_add,
    const T& m1_add,
    const T& m2_add,
    int64_t& m0,
    T& m1,
    T& m2) {
  const int64_t n = m0 + m0_add;
  const T c =
      n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
  const T delta = m1_add - m1;
  m1 += c * delta;
  m2 += m2_add + delta * delta * c * static_cast<T>(m0);
  m0 = n;
}

template <typename T>
ET_INLINE void AddMomentsVec(
    int64_t m0_add,
    const executorch::vec::Vectorized<T>& m1_add,
    const executorch::vec::Vectorized<T>& m2_add,
    int64_t& m0,
    executorch::vec::Vectorized<T>& m1,
    executorch::vec::Vectorized<T>& m2) {
  using Vec = executorch::vec::Vectorized<T>;
  const int64_t n = m0 + m0_add;
  const T c =
      n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
  const Vec c_vec(c);
  const Vec delta = m1_add - m1;
  m1 += c_vec * delta;
  m2 += m2_add + delta * delta * c_vec * Vec(static_cast<T>(m0));
  m0 = n;
}

template <typename T>
inline void UpdateMomentsVec(
    int64_t m0,
    const T* X_ptr,
    const std::array<executorch::vec::Vectorized<acc_t<T>>, kChunkSize>& c_vecs,
    int64_t& m0_stk0,
    executorch::vec::Vectorized<acc_t<T>>& m1_stk0,
    executorch::vec::Vectorized<acc_t<T>>& m2_stk0) {
  using Vec = executorch::vec::Vectorized<acc_t<T>>;
  Vec m1_vec(0);
  Vec m2_vec(0);
  for (int64_t j = 0; j < m0; ++j) {
    const Vec x_vec = Vec::loadu(X_ptr + j * Vec::size());
    const Vec delta_vec = x_vec - m1_vec;
    m1_vec += delta_vec * c_vecs[j];
    m2_vec += delta_vec * (x_vec - m1_vec);
  }
  AddMomentsVec(m0, m1_vec, m2_vec, m0_stk0, m1_stk0, m2_stk0);
}

// Compute rowwise moments by parallel Welford algorithm and cascade sum to
// improve numerical stability.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
// https://en.wikipedia.org/wiki/Pairwise_summation
template <typename T, int64_t kMaxDepth>
std::pair<acc_t<T>, acc_t<T>>
RowwiseMomentsImpl(const T* X, int64_t N, int64_t ddof = 0) {
  using T_ACC = acc_t<T>;

  constexpr int64_t kVecSize = executorch::vec::Vectorized<T>::size();
  constexpr int64_t kAccVecSize = executorch::vec::Vectorized<T_ACC>::size();
  const int64_t n = N / kVecSize;
  const int64_t m = executorch::utils::divup(n, kChunkSize);
  const int64_t depth = executorch::utils::CeilLog2(m);

  using Vec = executorch::vec::Vectorized<T_ACC>;
  const Vec kZeroVec(T_ACC(0));
  std::array<int64_t, kMaxDepth> m0_stk;
  std::array<Vec, kMaxDepth> m1_stk;
  std::array<Vec, kMaxDepth> m2_stk;
  for (int64_t i = 0; i < kMaxDepth; ++i) {
    m0_stk[i] = 0;
    m1_stk[i] = kZeroVec;
    m2_stk[i] = kZeroVec;
  }

  for (int64_t i = 0; i < m; ++i) {
    const T* X_ptr = X + i * kChunkSize * kVecSize;
    const int64_t m0 = std::min(kChunkSize, n - i * kChunkSize);
    static std::array<Vec, kChunkSize> c_vecs = ([]() {
      std::array<Vec, kChunkSize> result;
      for (int64_t j = 0; j < kChunkSize; ++j) {
        result[j] = Vec(T_ACC(1) / static_cast<T_ACC>(j + 1));
      }
      return result;
    })();
    UpdateMomentsVec(m0, X_ptr, c_vecs, m0_stk[0], m1_stk[0], m2_stk[0]);

    int64_t mask = i + 1;
    for (int64_t j = 1; j < depth && (mask & 1) == 0; ++j) {
      AddMomentsVec(
          m0_stk[j - 1],
          m1_stk[j - 1],
          m2_stk[j - 1],
          m0_stk[j],
          m1_stk[j],
          m2_stk[j]);
      m0_stk[j - 1] = 0;
      m1_stk[j - 1] = kZeroVec;
      m2_stk[j - 1] = kZeroVec;
      mask >>= 1;
    }
  }
  for (int64_t i = 1; i < depth; ++i) {
    AddMomentsVec(
        m0_stk[i], m1_stk[i], m2_stk[i], m0_stk[0], m1_stk[0], m2_stk[0]);
  }

  std::array<T_ACC, kAccVecSize> m1_arr{};
  std::array<T_ACC, kAccVecSize> m2_arr{};
  m1_stk[0].store(m1_arr.data());
  m2_stk[0].store(m2_arr.data());

  int64_t m0 = 0;
  T_ACC m1 = 0;
  T_ACC m2 = 0;
  for (int64_t i = n * kVecSize; i < N; ++i) {
    T_ACC x = static_cast<T_ACC>(X[i]);
    const T_ACC delta = x - m1;
    ++m0;
    m1 += delta / static_cast<T_ACC>(m0);
    m2 += delta * (x - m1);
  }
  // for BFloat16, each vector in m1_arr/m2_arr holds 2*n accumulated result
  int64_t m0_add = n * kVecSize / kAccVecSize;
  for (int64_t i = 0; i < kAccVecSize; ++i) {
    AddMoments(m0_add, m1_arr[i], m2_arr[i], m0, m1, m2);
  }

  return std::make_pair(m1, m2 / static_cast<T_ACC>(N - ddof));
}

template <typename T>
std::pair<acc_t<T>, acc_t<T>>
RowwiseMoments(const T* X, int64_t N, int64_t ddof = 0) {
  using Vec = executorch::vec::Vectorized<T>;
  constexpr int64_t kVecSize = Vec::size();
  const int64_t n = N / kVecSize;
  const int64_t m = executorch::utils::divup(n, kChunkSize);
  const int64_t depth = executorch::utils::CeilLog2(m);
  if (depth <= 4) {
    return RowwiseMomentsImpl<T, 4>(X, N, ddof);
  } else if (depth <= 8) {
    return RowwiseMomentsImpl<T, 8>(X, N, ddof);
  } else if (depth <= 16) {
    return RowwiseMomentsImpl<T, 16>(X, N, ddof);
  } else if (depth <= 32) {
    return RowwiseMomentsImpl<T, 32>(X, N, ddof);
  } else {
    return RowwiseMomentsImpl<T, 64>(X, N, ddof);
  }
}

} // namespace native
} // namespace executor
} // namespace torch
