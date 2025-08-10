/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#if defined(ET_USE_PYTORCH_HEADERS) && ET_USE_PYTORCH_HEADERS
#include <ATen/cpu/vec/vec.h>
#endif

#include <type_traits>

namespace torch {
namespace executor {
namespace native {
namespace utils {

/**
 * Python's __floordiv__ operator is more complicated than just floor(a / b).
 * It aims to maintain the property: a == (a // b) * b + remainder(a, b)
 * which can otherwise fail due to rounding errors in the remainder.
 * So, instead it is calculated as: a // b = (a - remainder(a, b)) / b
 * With some additional fix-ups added to the result.
 */
template <
    typename INT_T,
    typename std::enable_if<std::is_integral<INT_T>::value, bool>::type = true>
INT_T floor_divide(INT_T a, INT_T b) {
  const auto quot = a / b;
  // MSVC does not like signbit on integral types.
  if ((a < 0) == (b < 0)) {
    return quot;
  }
  const auto rem = a % b;
  return rem ? quot - 1 : quot;
}

template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
FLOAT_T floor_divide(FLOAT_T a, FLOAT_T b) {
  if (b == 0) {
    return std::signbit(a) ? -INFINITY : INFINITY;
  }
  const auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && std::signbit(b) != std::signbit(mod)) {
    return div - 1;
  }
  return div;
}

/**
 * A wrapper around std::isnan that works with MSVC. When building with MSVC,
 * std::isnan calls with integer inputs fail to compile due to ambiguous
 * overload resolution.
 */
template <typename T>
bool isnan_override(T a) {
  if constexpr (!std::is_integral_v<T>) {
    return std::isnan(a);
  } else {
    return false;
  }
}

/**
 * Override min/max so we can emulate PyTorch's behavior with NaN entries.
 */

template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
FLOAT_T min_override(FLOAT_T a, FLOAT_T b) {
  if (std::isnan(a)) {
    return a;
  } else if (std::isnan(b)) {
    return b;
  } else {
    return std::min(a, b);
  }
}

template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
FLOAT_T max_override(FLOAT_T a, FLOAT_T b) {
  if (std::isnan(a)) {
    return a;
  } else if (std::isnan(b)) {
    return b;
  } else {
    return std::max(a, b);
  }
}

template <
    typename INT_T,
    typename std::enable_if<std::is_integral<INT_T>::value, bool>::type = true>
INT_T min_override(INT_T a, INT_T b) {
  return std::min(a, b);
}

template <
    typename INT_T,
    typename std::enable_if<std::is_integral<INT_T>::value, bool>::type = true>
INT_T max_override(INT_T a, INT_T b) {
  return std::max(a, b);
}

template <
    typename T,
    typename std::enable_if_t<
        std::is_same_v<T, executorch::aten::Half> ||
            std::is_same_v<T, executorch::aten::BFloat16>,
        bool> = true>
T min_override(T a, T b) {
  const auto float_a = static_cast<float>(a);
  if (std::isnan(float_a)) {
    return a;
  }
  const auto float_b = static_cast<float>(b);
  if (std::isnan(float_b)) {
    return b;
  }

  if (float_a < float_b) {
    return a;
  }
  return b;
}

template <
    typename T,
    typename std::enable_if_t<
        std::is_same_v<T, executorch::aten::Half> ||
            std::is_same_v<T, executorch::aten::BFloat16>,
        bool> = true>
T max_override(T a, T b) {
  const auto float_a = static_cast<float>(a);
  if (std::isnan(float_a)) {
    return a;
  }
  const auto float_b = static_cast<float>(b);
  if (std::isnan(float_b)) {
    return b;
  }

  if (float_a > float_b) {
    return a;
  }
  return b;
}

#if defined(ET_USE_PYTORCH_HEADERS) && ET_USE_PYTORCH_HEADERS
template <typename T>
at::vec::Vectorized<T> min_override(
    at::vec::Vectorized<T> a,
    at::vec::Vectorized<T> b) {
  return at::vec::minimum(a, b);
}

template <typename T>
at::vec::Vectorized<T> min_override(at::vec::Vectorized<T> a, T b) {
  return min_override(a, at::vec::Vectorized<T>(b));
}

template <typename T>
at::vec::Vectorized<T> max_override(
    at::vec::Vectorized<T> a,
    at::vec::Vectorized<T> b) {
  return at::vec::maximum(a, b);
}

template <typename T>
at::vec::Vectorized<T> max_override(at::vec::Vectorized<T> a, T b) {
  return max_override(a, at::vec::Vectorized<T>(b));
}

#endif
/**
 * There is a slight difference in how std::fmod works compared to how ATen
 * determines remainders:
 * The returned value of std::fmod has the same sign as x and is less than y in
 * magnitude. (https://en.cppreference.com/w/cpp/numeric/math/fmod)
 * On the other hand, ATen's remainder always matches the sign of y
 * To correct this, we need to add y to the remainder when one but not both of
 * x and y is negative and the remainder is not 0
 */

template <
    typename CTYPE,
    typename std::enable_if<std::is_floating_point<CTYPE>::value, int>::type =
        0>
CTYPE remainder_override(CTYPE a, CTYPE b) {
  float rem = std::fmod(a, b);
  if (((a < 0) ^ (b < 0)) && rem != 0) {
    rem += b;
  }
  return rem;
}

template <
    typename CTYPE,
    typename std::enable_if<std::is_integral<CTYPE>::value, int>::type = 0>
CTYPE remainder_override(CTYPE a, CTYPE b) {
  return a % b;
}

} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
