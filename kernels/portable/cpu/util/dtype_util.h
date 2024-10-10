/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace utils {
namespace internal {

template <typename To, typename From>
To load_and_convert(const void* fromPtr) {
  return static_cast<To>(*reinterpret_cast<const From*>(fromPtr));
}

template <typename To, typename From>
void convert_and_store(From f, void* dst) {
  *reinterpret_cast<To*>(dst) = static_cast<To>(f);
}

template <typename CTYPE_COMMON>
using load_to_common_fn = CTYPE_COMMON (*)(const void*);

template <typename CTYPE_COMMON, const char* op_name>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn_realhbbf16(
    const Tensor& t) {
  CTYPE_COMMON (*result)(const void*) = nullptr;
  ET_SWITCH_REALHBBF16_TYPES(
      t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::load_and_convert<CTYPE_COMMON, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn_realhbf16(
    const Tensor& t) {
  CTYPE_COMMON (*result)(const void*) = nullptr;
  ET_SWITCH_REALHBF16_TYPES(
      t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::load_and_convert<CTYPE_COMMON, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn_floathbf16(
    const Tensor& t) {
  CTYPE_COMMON (*result)(const void*) = nullptr;
  ET_SWITCH_FLOATHBF16_TYPES(
      t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::load_and_convert<CTYPE_COMMON, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn_intb(const Tensor& t) {
  CTYPE_COMMON (*result)(const void*) = nullptr;
  ET_SWITCH_INT_TYPES_AND(
      Bool, t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::load_and_convert<CTYPE_COMMON, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn_bool_or_byte(
    const Tensor& t) {
  CTYPE_COMMON (*result)(const void*) = nullptr;
  ET_SWITCH_TWO_TYPES(
      Bool, Byte, t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::load_and_convert<CTYPE_COMMON, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn_same_as_compute(
    const Tensor& t) {
  constexpr auto common_scalar_type = CppTypeToScalarType<CTYPE_COMMON>::value;
  ET_CHECK_MSG(
      t.scalar_type() == common_scalar_type,
      "Unhandled dtype %s for %s",
      ::executorch::runtime::toString(common_scalar_type),
      op_name);
  return internal::load_and_convert<CTYPE_COMMON, CTYPE_COMMON>;
}

template <
    typename CTYPE_COMMON,
    const char* op_name,
    std::enable_if_t<std::is_same_v<CTYPE_COMMON, float>, bool> = true>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn_same_as_common(
    const Tensor& t) {
  CTYPE_COMMON (*result)(const void*) = nullptr;
  ET_SWITCH_THREE_TYPES(
      Float, Half, BFloat16, t.scalar_type(), unused, op_name, T, [&]() {
        result = internal::load_and_convert<CTYPE_COMMON, T>;
      });
  return result;
}

template <
    typename CTYPE_COMMON,
    const char* op_name,
    std::enable_if_t<!std::is_same_v<CTYPE_COMMON, float>, bool> = true>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn_same_as_common(
    const Tensor& t) {
  return get_load_to_common_fn_same_as_compute<CTYPE_COMMON, op_name>(t);
}

template <typename CTYPE_COMMON>
using store_common_to_tensor_fn = void (*)(CTYPE_COMMON, void*);

template <typename CTYPE_COMMON, const char* op_name>
store_common_to_tensor_fn<CTYPE_COMMON>
get_store_common_to_tensor_fn_realhbbf16(const Tensor& t) {
  void (*result)(CTYPE_COMMON, void*) = nullptr;
  ET_SWITCH_REALHBBF16_TYPES(
      t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMMON>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
store_common_to_tensor_fn<CTYPE_COMMON> get_store_common_to_tensor_fn_realhbf16(
    const Tensor& t) {
  void (*result)(CTYPE_COMMON, void*) = nullptr;
  ET_SWITCH_REALHBF16_TYPES(
      t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMMON>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
store_common_to_tensor_fn<CTYPE_COMMON>
get_store_common_to_tensor_fn_floathbf16(const Tensor& t) {
  void (*result)(CTYPE_COMMON, void*) = nullptr;
  ET_SWITCH_FLOATHBF16_TYPES(
      t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMMON>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
store_common_to_tensor_fn<CTYPE_COMMON> get_store_common_to_tensor_fn_intb(
    const Tensor& t) {
  void (*result)(CTYPE_COMMON, void*) = nullptr;
  ET_SWITCH_INT_TYPES_AND(
      Bool, t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMMON>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
store_common_to_tensor_fn<CTYPE_COMMON>
get_store_common_to_tensor_fn_bool_or_byte(const Tensor& t) {
  void (*result)(CTYPE_COMMON, void*) = nullptr;
  ET_SWITCH_TWO_TYPES(
      Bool, Byte, t.scalar_type(), unused, op_name, TENSOR_CTYPE, [&]() {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMMON>;
      });
  return result;
}

template <typename CTYPE_COMMON, const char* op_name>
store_common_to_tensor_fn<CTYPE_COMMON>
get_store_common_to_tensor_fn_same_as_compute(const Tensor& t) {
  constexpr auto common_scalar_type = CppTypeToScalarType<CTYPE_COMMON>::value;
  ET_CHECK_MSG(
      t.scalar_type() == common_scalar_type,
      "Unhandled dtype %s for %s",
      ::executorch::runtime::toString(common_scalar_type),
      op_name);
  return internal::convert_and_store<CTYPE_COMMON, CTYPE_COMMON>;
}

template <
    typename CTYPE_COMMON,
    const char* op_name,
    std::enable_if_t<std::is_same_v<CTYPE_COMMON, float>, bool> = true>
store_common_to_tensor_fn<CTYPE_COMMON>
get_store_common_to_tensor_fn_same_as_common(const Tensor& t) {
  void (*result)(CTYPE_COMMON, void*) = nullptr;
  ET_SWITCH_THREE_TYPES(
      Float, Half, BFloat16, t.scalar_type(), unused, op_name, CTYPE, [&]() {
        result = internal::convert_and_store<CTYPE, CTYPE_COMMON>;
      });
  return result;
}

template <
    typename CTYPE_COMMON,
    const char* op_name,
    std::enable_if_t<!std::is_same_v<CTYPE_COMMON, float>, bool> = true>
store_common_to_tensor_fn<CTYPE_COMMON>
get_store_common_to_tensor_fn_same_as_common(const Tensor& t) {
  return get_store_common_to_tensor_fn_same_as_compute<CTYPE_COMMON, op_name>(
      t);
}

} // namespace internal

enum class SupportedTensorDtypes {
  REALHBBF16,
  REALHBF16,
  FLOATHBF16,
  INTB,
  BOOL_OR_BYTE,
  SAME_AS_COMPUTE,
  SAME_AS_COMMON,
};

namespace internal {

template <typename CTYPE_COMMON, const char* op_name>
load_to_common_fn<CTYPE_COMMON> get_load_to_common_fn(
    const Tensor& t,
    SupportedTensorDtypes dtypes) {
  switch (dtypes) {
    case SupportedTensorDtypes::REALHBBF16:
      return get_load_to_common_fn_realhbbf16<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::REALHBF16:
      return get_load_to_common_fn_realhbf16<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::FLOATHBF16:
      return get_load_to_common_fn_realhbf16<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::INTB:
      return get_load_to_common_fn_intb<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::BOOL_OR_BYTE:
      return get_load_to_common_fn_bool_or_byte<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::SAME_AS_COMPUTE:
      return get_load_to_common_fn_same_as_compute<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::SAME_AS_COMMON:
      return get_load_to_common_fn_same_as_common<CTYPE_COMMON, op_name>(t);
  }
  ET_CHECK(false);
  return nullptr;
}

template <typename CTYPE_COMMON, const char* op_name>
store_common_to_tensor_fn<CTYPE_COMMON> get_store_common_to_tensor_fn(
    const Tensor& t,
    SupportedTensorDtypes dtypes) {
  switch (dtypes) {
    case SupportedTensorDtypes::REALHBBF16:
      return get_store_common_to_tensor_fn_realhbbf16<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::REALHBF16:
      return get_store_common_to_tensor_fn_realhbf16<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::FLOATHBF16:
      return get_store_common_to_tensor_fn_floathbf16<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::INTB:
      return get_store_common_to_tensor_fn_intb<CTYPE_COMMON, op_name>(t);
    case SupportedTensorDtypes::BOOL_OR_BYTE:
      return get_store_common_to_tensor_fn_bool_or_byte<CTYPE_COMMON, op_name>(
          t);
    case SupportedTensorDtypes::SAME_AS_COMPUTE:
      return get_store_common_to_tensor_fn_same_as_compute<
          CTYPE_COMMON,
          op_name>(t);
    case SupportedTensorDtypes::SAME_AS_COMMON: {
      return get_store_common_to_tensor_fn_same_as_common<
          CTYPE_COMMON,
          op_name>(t);
    }
  }
  ET_CHECK(false);
  return nullptr;
}

bool check_tensor_dtype(
    const Tensor t,
    SupportedTensorDtypes dtypes,
    const ScalarType compute_type);

} // namespace internal
} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
