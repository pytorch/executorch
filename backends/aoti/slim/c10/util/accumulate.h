/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <executorch/runtime/platform/assert.h>

#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>

namespace executorch::backends::aoti::slim::c10 {

/// Sum of a list of integers; accumulates into the int64_t datatype
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t sum_integers(const C& container) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      container.begin(), container.end(), static_cast<int64_t>(0));
}

/// Sum of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    std::enable_if_t<
        std::is_integral_v<typename std::iterator_traits<Iter>::value_type>,
        int> = 0>
inline int64_t sum_integers(Iter begin, Iter end) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(begin, end, static_cast<int64_t>(0));
}

/// Product of a list of integers; accumulates into the int64_t datatype
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t multiply_integers(const C& container) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      container.begin(),
      container.end(),
      static_cast<int64_t>(1),
      std::multiplies<>());
}

/// Product of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    std::enable_if_t<
        std::is_integral_v<typename std::iterator_traits<Iter>::value_type>,
        int> = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      begin, end, static_cast<int64_t>(1), std::multiplies<>());
}

/// Return product of all dimensions starting from k
/// Returns 1 if k>=dims.size()
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_from_dim(const int k, const C& dims) {
  ET_DCHECK_MSG(k >= 0, "numelements_from_dim: k must be non-negative");

  if (k > static_cast<int>(dims.size())) {
    return 1;
  } else {
    auto cbegin = dims.cbegin();
    std::advance(cbegin, k);
    return multiply_integers(cbegin, dims.cend());
  }
}

/// Product of all dims up to k (not including dims[k])
/// Throws an error if k>dims.size()
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_to_dim(const int k, const C& dims) {
  ET_CHECK_MSG(0 <= k, "numelements_to_dim: k must be non-negative");
  ET_CHECK_MSG(
      (unsigned)k <= dims.size(),
      "numelements_to_dim: k must not exceed dims.size()");

  auto cend = dims.cbegin();
  std::advance(cend, k);
  return multiply_integers(dims.cbegin(), cend);
}

/// Product of all dims between k and l (including dims[k] and excluding
/// dims[l]) k and l may be supplied in either order
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_between_dim(int k, int l, const C& dims) {
  ET_CHECK_MSG(0 <= k, "numelements_between_dim: k must be non-negative");
  ET_CHECK_MSG(0 <= l, "numelements_between_dim: l must be non-negative");

  if (k > l) {
    std::swap(k, l);
  }

  ET_CHECK_MSG(
      (unsigned)l < dims.size(),
      "numelements_between_dim: l must be less than dims.size()");

  auto cbegin = dims.cbegin();
  auto cend = dims.cbegin();
  std::advance(cbegin, k);
  std::advance(cend, l);
  return multiply_integers(cbegin, cend);
}

} // namespace executorch::backends::aoti::slim::c10
