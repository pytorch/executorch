/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

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
  if (std::signbit(a) == std::signbit(b)) {
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
    typename std::enable_if<std::is_same<T, exec_aten::Half>::value, bool>::
        type = true>
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
    typename std::enable_if<std::is_same<T, exec_aten::Half>::value, bool>::
        type = true>
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
