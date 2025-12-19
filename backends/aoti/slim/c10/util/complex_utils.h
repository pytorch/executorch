/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if !defined(STANDALONE_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "standalone/c10/util/complex_utils.h is not meant to be individually included. Include standalone/c10/util/complex.h instead."
#endif

#include <limits>

namespace executorch::backends::aoti::slim::c10 {

template <typename T>
struct is_complex : public std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

template <typename T>
struct is_complex<executorch::backends::aoti::slim::c10::complex<T>>
    : public std::true_type {};

// Extract double from std::complex<double>; is identity otherwise
// TODO: Write in more idiomatic C++17
template <typename T>
struct scalar_value_type {
  using type = T;
};
template <typename T>
struct scalar_value_type<std::complex<T>> {
  using type = T;
};
template <typename T>
struct scalar_value_type<executorch::backends::aoti::slim::c10::complex<T>> {
  using type = T;
};

} // namespace executorch::backends::aoti::slim::c10

namespace std {

template <typename T>
class numeric_limits<executorch::backends::aoti::slim::c10::complex<T>>
    : public numeric_limits<T> {};

template <typename T>
bool isnan(const executorch::backends::aoti::slim::c10::complex<T>& v) {
  return std::isnan(v.real()) || std::isnan(v.imag());
}

} // namespace std
