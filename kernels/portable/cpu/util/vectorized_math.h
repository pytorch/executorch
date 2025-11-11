/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#if defined(ET_USE_PYTORCH_HEADERS) && ET_USE_PYTORCH_HEADERS
#include <ATen/cpu/vec/vec.h>
#endif // ET_USE_PYTORCH_HEADERS

#include <iostream>
#include <type_traits>

#if defined(ET_USE_PYTORCH_HEADERS) && ET_USE_PYTORCH_HEADERS
namespace executorch {
inline namespace math {
namespace internal {
template <typename T>
auto convert_to_vectorized_n_of_float(at::vec::Vectorized<T> vec) {
  static constexpr auto float_vec_size = at::vec::Vectorized<float>::size();
  static constexpr auto t_vec_size = at::vec::Vectorized<T>::size();
  static constexpr auto result_size =
      t_vec_size < float_vec_size ? 1 : t_vec_size / float_vec_size;
  static_assert(result_size >= 1);
  return at::vec::convert<float, result_size, T, 1, /*keep=*/true>(
      at::vec::VectorizedN<T, 1>(vec));
}
} // namespace internal
} // namespace math
} // namespace executorch
#endif // ET_USE_PYTORCH_HEADERS

#define _ET_INTERNAL_STD_MATH_FUNC(name) \
  namespace executorch {                 \
  inline namespace math {                \
  using std::name;                       \
  }                                      \
  } // namespace executorch

#if defined(ET_USE_PYTORCH_HEADERS) && ET_USE_PYTORCH_HEADERS
/**
 * Internal-usage macro for making a vectorized variant of a unary
 * function available in the executorch::math namespace.
 */
#define ET_INTERNAL_VECTORIZED_FLOAT_UNARY_FUNC(func_name)                \
  namespace executorch {                                                  \
  inline namespace math {                                                 \
  template <typename T>                                                   \
  auto func_name(at::vec::Vectorized<T> vec) {                            \
    if constexpr (!::executorch::runtime::is_floating_point<T>::value) {  \
      return internal::convert_to_vectorized_n_of_float(vec).func_name(); \
    } else {                                                              \
      return vec.func_name();                                             \
    }                                                                     \
  }                                                                       \
  }                                                                       \
  }

#define ET_INTERNAL_VECTORIZED_FLOAT_BINARY_FUNC(func_name)                  \
  namespace executorch {                                                     \
  inline namespace math {                                                    \
  template <typename T>                                                      \
  auto func_name(at::vec::Vectorized<T> vec0, at::vec::Vectorized<T> vec1) { \
    if constexpr (!::executorch::runtime::is_floating_point<T>::value) {     \
      const auto vec_float0 =                                                \
          internal::convert_to_vectorized_n_of_float(vec0);                  \
      const auto vec_float1 =                                                \
          internal::convert_to_vectorized_n_of_float(vec1);                  \
      return vec_float0.func_name(vec_float1);                               \
    } else {                                                                 \
      return vec0.func_name(vec1);                                           \
    }                                                                        \
  }                                                                          \
  }                                                                          \
  }

/**
 * Internal-usage macro for making a C++ standard library
 * floating-point function and a vectorized variant of it available in
 * the c10::math namespace. Should be used with functions where the
 * corresponding operator is a "float op" in TensorIterator parlance
 * (i.e., uses something like build_borrowing_binary_float_op()),
 * because it converts non-floating-point arguments to floating point.
 */
#define ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(func_name) \
  _ET_INTERNAL_STD_MATH_FUNC(func_name)                        \
  ET_INTERNAL_VECTORIZED_FLOAT_UNARY_FUNC(func_name)

#define ET_INTERNAL_VECTORIZED_STD_FLOAT_BINARY_FUNC(func_name) \
  _ET_INTERNAL_STD_MATH_FUNC(func_name)                         \
  ET_INTERNAL_VECTORIZED_FLOAT_BINARY_FUNC(func_name)

#else // ET_USE_PYTORCH_HEADERS
#define ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(name) \
  _ET_INTERNAL_STD_MATH_FUNC(name)
#define ET_INTERNAL_VECTORIZED_STD_FLOAT_BINARY_FUNC(name) \
  _ET_INTERNAL_STD_MATH_FUNC(name)
#endif // ET_USE_PYTORCH_HEADERS

// To simplify client code, we provide coverage for a bunch of float ops (the
// same ones listed in ATen vml.h) here.
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(abs)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(acos)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(asin)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(atan)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(ceil)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(cos)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(cosh)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(erf)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(erfc)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(exp)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(expm1)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(floor)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(log)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(log10)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(log1p)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(log2)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(sin)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(sinh)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(sqrt)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(round)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(tan)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(tanh)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(trunc)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(lgamma)

#if defined(ET_USE_PYTORCH_HEADERS) && ET_USE_PYTORCH_HEADERS
ET_INTERNAL_VECTORIZED_FLOAT_BINARY_FUNC(rsqrt)
#endif // ET_USE_PYTORCH_HEADERS

namespace executorch {
inline namespace math {
template <typename T, std::enable_if_t<std::is_floating_point_v<T>>>
T rsqrt(T x) {
  return T(1) / std::sqrt(x);
}
} // namespace math
} // namespace executorch

ET_INTERNAL_VECTORIZED_STD_FLOAT_BINARY_FUNC(atan2)
ET_INTERNAL_VECTORIZED_STD_FLOAT_BINARY_FUNC(fmod)
ET_INTERNAL_VECTORIZED_STD_FLOAT_BINARY_FUNC(pow)
