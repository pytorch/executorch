/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#ifdef ET_USE_PYTORCH_HEADERS
#include <ATen/cpu/vec/vec.h>
#endif // ET_USE_PYTORCH_HEADERS

#define _ET_INTERNAL_STD_MATH_FUNC(name) \
  namespace executorch {                 \
  inline namespace math {                \
  using std::name;                       \
  }                                      \
  } // namespace executorch

#ifdef ET_USE_PYTORCH_HEADERS
/**
 * Internal-usage macro for making a vectorized variant of a unary
 * function available in the executorch::math namespace.
 */
#define ET_INTERNAL_VECTORIZED_FLOAT_UNARY_FUNC(func_name)               \
  namespace executorch {                                                 \
  inline namespace math {                                                \
  template <typename T>                                                  \
  auto func_name(at::vec::Vectorized<T> vec) {                           \
    if constexpr (!::executorch::runtime::is_floating_point<T>::value) { \
      return at::vec::convert<float>(vec).func_name();                   \
    } else {                                                             \
      return vec.func_name();                                            \
    }                                                                    \
  }                                                                      \
  }                                                                      \
  }

#define ET_INTERNAL_VECTORIZED_FLOAT_BINARY_FUNC(func_name)                  \
  namespace executorch {                                                     \
  inline namespace math {                                                    \
  template <typename T>                                                      \
  auto func_name(at::vec::Vectorized<T> vec0, at::vec::Vectorized<T> vec1) { \
    if constexpr (!::executorch::runtime::is_floating_point<T>::value) {     \
      return at::vec::convert<float>(vec0).func_name(                        \
          at::vec::convert<float>(vec1));                                    \
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
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(rsqrt)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(tan)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(tanh)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(trunc)
ET_INTERNAL_VECTORIZED_STD_FLOAT_UNARY_FUNC(lgamma)

ET_INTERNAL_VECTORIZED_STD_FLOAT_BINARY_FUNC(atan2)
ET_INTERNAL_VECTORIZED_STD_FLOAT_BINARY_FUNC(fmod)
ET_INTERNAL_VECTORIZED_STD_FLOAT_BINARY_FUNC(pow)
