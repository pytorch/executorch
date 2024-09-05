/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <executorch/kernels/portable/cpu/selective_build.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/portable_type/scalar.h>

#define ET_CHECK_SCALAR_SAME_TYPE(a__, b__)                      \
  ({                                                             \
    ET_CHECK_MSG(                                                \
        (a__).isBoolean() == (b__).isBoolean(),                  \
        "Scalars type do not match, isBoolean() %d vs %d",       \
        (a__).isBoolean(),                                       \
        (b__).isBoolean());                                      \
    ET_CHECK_MSG(                                                \
        (a__).isIntegral(false) == (b__).isIntegral(false),      \
        "Scalars type do not match, isIntegral(false) %d vs %d", \
        (a__).isIntegral(false),                                 \
        (b__).isIntegral(false));                                \
    ET_CHECK_MSG(                                                \
        (a__).isFloatingPoint() == (b__).isFloatingPoint(),      \
        "Scalars type do not match, isFloatingPoint() %d vs %d", \
        (a__).isFloatingPoint(),                                 \
        (b__).isFloatingPoint());                                \
  })

/**
 * Convenience macro to extract a Scalar into a value
 */
#define ET_EXTRACT_SCALAR(scalar, out_val)     \
  ET_CHECK_MSG(                                \
      utils::extract_scalar(scalar, &out_val), \
      #scalar " could not be extracted: wrong type or out of range");

namespace torch {
namespace executor {
namespace native {
namespace utils {

/**
 * Returns the dtype associated with a Scalar that reflects the category
 * of value stored by the Scalar.
 */
inline ScalarType get_scalar_dtype(Scalar scalar) {
  if (scalar.isBoolean()) {
    return ScalarType::Bool;
  }
  if (scalar.isIntegral(false)) {
    return ScalarType::Long;
  }
  if (scalar.isFloatingPoint()) {
    return ScalarType::Double;
  }
  ET_CHECK_MSG(false, "Scalar must be Boolean, Integral or Floating.");
}

inline bool scalars_have_same_dtype(Scalar a, Scalar b) {
  ScalarType a_dtype = get_scalar_dtype(a);
  ScalarType b_dtype = get_scalar_dtype(b);
  if (a_dtype == b_dtype) {
    return true;
  }
  ET_LOG(
      Error,
      "Expected scalars to have the same dtype, but found %s and %s",
      toString(a_dtype),
      toString(b_dtype));
  return false;
}

template <typename T1, typename T2, bool half_to_float = false>
struct promote_type_with_scalar_type {
 private:
  static_assert(
      std::is_same<T2, torch::executor::internal::B1>::value ||
          std::is_same<T2, torch::executor::internal::I8>::value ||
          std::is_same<T2, torch::executor::internal::F8>::value,
      "scalar type can only be Bool, Long or Double");
  static_assert(
      !is_qint_type<T1>::value,
      "promote_type_with_scalar_type not valid for quantized dtypes");
  static_assert(
      !is_bits_type<T1>::value,
      "promote_type_with_scalar_type not valid for bits dtypes");
  using promote_type_with_scalar_type_not_respecting_half_to_float =
      typename std::conditional<
          is_complex_type<T1>::value ||
              std::is_same<T2, torch::executor::internal::B1>::value,
          T1,
          typename std::conditional<
              std::is_same<T2, torch::executor::internal::I8>::value,
              typename std::conditional<
                  std::is_same<T1, torch::executor::internal::B1>::value,
                  torch::executor::internal::I8,
                  T1>::type,
              typename std::conditional<
                  is_floating_point<T1>::value,
                  T1,
                  torch::executor::internal::F4>::type>::type>::type;

 public:
  using type = typename std::conditional<
      half_to_float &&
          (std::is_same<
               promote_type_with_scalar_type_not_respecting_half_to_float,
               typename ScalarTypeToCppType<
                   exec_aten::ScalarType::Half>::type>::value ||
           std::is_same<
               promote_type_with_scalar_type_not_respecting_half_to_float,
               typename ScalarTypeToCppType<
                   exec_aten::ScalarType::BFloat16>::type>::value),
      typename ScalarTypeToCppType<exec_aten::ScalarType::Float>::type,
      promote_type_with_scalar_type_not_respecting_half_to_float>::type;
};

/**
 * Implement type promotion between a tensor's ScalarType with a Scalar.
 * If the Scalar contains a value in the same category of the tensor's
 * ScalarType, the tensor's ScalarType will be preserved. Otherwise, a type
 * promotion will occur and the dtype associated with the Scalar will be
 * returned.
 *
 * If t is a complex type, then it will be preserved.
 */
inline ScalarType promote_type_with_scalar(
    ScalarType t,
    Scalar scalar,
    bool half_to_float = false) {
  if (half_to_float && t == ScalarType::Half) {
    t = ScalarType::Float;
  }

  // QInt, and Bits types not supported
  ET_CHECK(!isQIntType(t));
  ET_CHECK(!isBitsType(t));

  if (isComplexType(t)) {
    return t;
  }
  if (scalar.isFloatingPoint()) {
    if (isFloatingType(t)) {
      return t;
    } else {
      // ATen will promote to Float instead of Double
      return ScalarType::Float;
    }
  }
  if (scalar.isIntegral(false)) {
    if (isFloatingType(t) || isIntegralType(t, false)) {
      return t;
    } else {
      return ScalarType::Long;
    }
  }
  if (scalar.isBoolean()) {
    return t;
  }
  ET_CHECK_MSG(false, "Scalar must be Boolean, Integral or Floating.");
}

/**
 * Extracts an integer value from a Scalar.
 *
 * @param[in] scalar The source of the value to extract.
 * @param[out] out_val The extracted value, on success.
 * @returns `true` if a value was extracted, and sets `*out_val` to that value.
 *    `false` if a value could not be extracted: either it was not an integer
 *    Scalar, or the value of that Scalar could not be represented by INT_T.
 */
template <
    typename INT_T,
    typename std::enable_if<
        std::is_integral<INT_T>::value && !std::is_same<INT_T, bool>::value,
        bool>::type = true>
bool extract_scalar(Scalar scalar, INT_T* out_val) {
  if (!scalar.isIntegral(/*includeBool=*/false)) {
    return false;
  }
  int64_t val = scalar.to<int64_t>();
  if (val < std::numeric_limits<INT_T>::lowest() ||
      val > std::numeric_limits<INT_T>::max()) {
    // PyTorch's implementation of clamp() raises an exception if the min/max
    // values cannot be represented as the dtype, so we should fail too.
    return false;
  }
  *out_val = static_cast<INT_T>(val);
  return true;
}

/**
 * Extracts a floating point value from a Scalar.
 *
 * @param[in] scalar The source of the value to extract.
 * @param[out] out_val The extracted value, on success.
 * @returns `true` if a value was extracted, and sets `*out_val` to that value.
 *    `false` if a value could not be extracted: either it was not a floating
 *    point Scalar, or the value of that Scalar could not be represented by
 *    FLOAT_T.
 */
template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
bool extract_scalar(Scalar scalar, FLOAT_T* out_val) {
  double val;
  if (scalar.isFloatingPoint()) {
    val = scalar.to<double>();
    // If the double is outside the finite range supported by float, it cannot
    // be represented when FLOAT_T == float. float can, however, represent
    // infinite and NaN values.
    if (std::isfinite(val) &&
        (val < std::numeric_limits<FLOAT_T>::lowest() ||
         val > std::numeric_limits<FLOAT_T>::max())) {
      // PyTorch's implementation of clamp() raises an exception if the min/max
      // values cannot be represented as the dtype, so we should fail too.
      return false;
    }
  } else if (scalar.isIntegral(/*includeBool=*/false)) {
    val = static_cast<double>(scalar.to<int64_t>());
  } else {
    // Not a numeric Scalar.
    return false;
  }
  *out_val = static_cast<FLOAT_T>(val);
  return true;
}

/**
 * Extracts a boolean value from a Scalar.
 *
 * @param[in] scalar The source of the value to extract.
 * @param[out] out_val The extracted value, on success.
 * @returns `true` if a value was extracted, and sets `*out_val` to that value.
 *    `false` if a value could not be extracted, i.e. not a boolean
 */
template <
    typename BOOL_T,
    typename std::enable_if<std::is_same<BOOL_T, bool>::value, bool>::type =
        true>
bool extract_scalar(Scalar scalar, BOOL_T* out_val) {
  if (scalar.isIntegral(false)) {
    *out_val = static_cast<bool>(scalar.to<int64_t>());
    return true;
  }
  if (scalar.isBoolean()) {
    *out_val = scalar.to<bool>();
    return true;
  }
  return false;
}

} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
