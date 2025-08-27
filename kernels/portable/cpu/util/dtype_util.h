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

template <typename CTYPE_COMPUTE>
using load_to_compute_fn = CTYPE_COMPUTE (*)(const void*);

template <typename CTYPE_COMPUTE, const char* op_name>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_realhbbf16(
    KernelRuntimeContext& context,
    const Tensor& t) {
  CTYPE_COMPUTE (*result)(const void*) = nullptr;
  ET_SWITCH_REALHBBF16_TYPES(
      t.scalar_type(), context, op_name, TENSOR_CTYPE, [&]() -> void {
        result = internal::load_and_convert<CTYPE_COMPUTE, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_realhbf16(
    KernelRuntimeContext& context,
    const Tensor& t) {
  CTYPE_COMPUTE (*result)(const void*) = nullptr;
  ET_SWITCH_REALHBF16_TYPES(
      t.scalar_type(), context, op_name, TENSOR_CTYPE, [&]() -> void {
        result = internal::load_and_convert<CTYPE_COMPUTE, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_floathbf16(
    KernelRuntimeContext& context,
    const Tensor& t) {
  CTYPE_COMPUTE (*result)(const void*) = nullptr;
  ET_SWITCH_FLOATHBF16_TYPES(
      t.scalar_type(), context, op_name, TENSOR_CTYPE, [&]() -> void {
        result = internal::load_and_convert<CTYPE_COMPUTE, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_intb(
    KernelRuntimeContext& context,
    const Tensor& t) {
  CTYPE_COMPUTE (*result)(const void*) = nullptr;
  ET_SWITCH_INT_TYPES_AND(
      Bool, t.scalar_type(), context, op_name, TENSOR_CTYPE, [&]() -> void {
        result = internal::load_and_convert<CTYPE_COMPUTE, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_bool(
    KernelRuntimeContext& context,
    const Tensor& t) {
  CTYPE_COMPUTE (*result)(const void*) = nullptr;
  if (t.scalar_type() != ScalarType::Bool) {
    context.fail(torch::executor::Error::InvalidArgument);
    ET_LOG(
        Error,
        "Unhandled dtype %s for %s",
        ::executorch::runtime::toString(t.scalar_type()),
        op_name);
  } else {
    result = internal::load_and_convert<CTYPE_COMPUTE, bool>;
  }
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_bool_or_byte(
    KernelRuntimeContext& context,
    const Tensor& t) {
  CTYPE_COMPUTE (*result)(const void*) = nullptr;
  ET_SWITCH_TWO_TYPES(
      Bool,
      Byte,
      t.scalar_type(),
      context,
      op_name,
      TENSOR_CTYPE,
      [&]() -> void {
        result = internal::load_and_convert<CTYPE_COMPUTE, TENSOR_CTYPE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_same_as_compute(
    KernelRuntimeContext& context,
    const Tensor& t) {
  CTYPE_COMPUTE (*result)(const void*) = nullptr;
  constexpr auto common_scalar_type = CppTypeToScalarType<CTYPE_COMPUTE>::value;
  if (t.scalar_type() != common_scalar_type) {
    context.fail(torch::executor::Error::InvalidArgument);
    ET_LOG(
        Error,
        "Unhandled dtype %s for %s",
        ::executorch::runtime::toString(t.scalar_type()),
        op_name);
  } else {
    result = internal::load_and_convert<CTYPE_COMPUTE, CTYPE_COMPUTE>;
  }
  return result;
}

template <
    typename CTYPE_COMPUTE,
    const char* op_name,
    std::enable_if_t<std::is_same_v<CTYPE_COMPUTE, float>, bool> = true>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_same_as_common(
    KernelRuntimeContext& context,
    const Tensor& t) {
  CTYPE_COMPUTE (*result)(const void*) = nullptr;
  ET_SWITCH_THREE_TYPES(
      Float,
      Half,
      BFloat16,
      t.scalar_type(),
      context,
      op_name,
      T,
      [&]() -> void { result = internal::load_and_convert<CTYPE_COMPUTE, T>; });
  return result;
}

template <
    typename CTYPE_COMPUTE,
    const char* op_name,
    std::enable_if_t<!std::is_same_v<CTYPE_COMPUTE, float>, bool> = true>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_same_as_common(
    KernelRuntimeContext& context,
    const Tensor& t) {
  return get_load_to_compute_fn_same_as_compute<CTYPE_COMPUTE, op_name>(
      context, t);
}

template <typename CTYPE_COMPUTE>
using store_compute_to_tensor_fn = void (*)(CTYPE_COMPUTE, void*);

template <typename CTYPE_COMPUTE, const char* op_name>
store_compute_to_tensor_fn<CTYPE_COMPUTE>
get_store_compute_to_tensor_fn_realhbbf16(
    KernelRuntimeContext& context,
    const Tensor& t) {
  void (*result)(CTYPE_COMPUTE, void*) = nullptr;
  ET_SWITCH_REALHBBF16_TYPES(
      t.scalar_type(), context, op_name, TENSOR_CTYPE, [&]() -> void {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
store_compute_to_tensor_fn<CTYPE_COMPUTE>
get_store_compute_to_tensor_fn_realhbf16(
    KernelRuntimeContext& context,
    const Tensor& t) {
  void (*result)(CTYPE_COMPUTE, void*) = nullptr;
  ET_SWITCH_REALHBF16_TYPES(
      t.scalar_type(), context, op_name, TENSOR_CTYPE, [&]() -> void {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
store_compute_to_tensor_fn<CTYPE_COMPUTE>
get_store_compute_to_tensor_fn_floathbf16(
    KernelRuntimeContext& context,
    const Tensor& t) {
  void (*result)(CTYPE_COMPUTE, void*) = nullptr;
  ET_SWITCH_FLOATHBF16_TYPES(
      t.scalar_type(), context, op_name, TENSOR_CTYPE, [&]() -> void {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
store_compute_to_tensor_fn<CTYPE_COMPUTE> get_store_compute_to_tensor_fn_intb(
    KernelRuntimeContext& context,
    const Tensor& t) {
  void (*result)(CTYPE_COMPUTE, void*) = nullptr;
  ET_SWITCH_INT_TYPES_AND(
      Bool, t.scalar_type(), context, op_name, TENSOR_CTYPE, [&]() -> void {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
store_compute_to_tensor_fn<CTYPE_COMPUTE> get_store_compute_to_tensor_fn_bool(
    KernelRuntimeContext& context,
    const Tensor& t) {
  void (*result)(CTYPE_COMPUTE, void*) = nullptr;
  if (t.scalar_type() != ScalarType::Bool) {
    context.fail(torch::executor::Error::InvalidArgument);
    ET_LOG(
        Error,
        "Unhandled dtype %s for %s",
        ::executorch::runtime::toString(t.scalar_type()),
        op_name);
  } else {
    result = internal::convert_and_store<bool, CTYPE_COMPUTE>;
  }
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
store_compute_to_tensor_fn<CTYPE_COMPUTE>
get_store_compute_to_tensor_fn_bool_or_byte(
    KernelRuntimeContext& context,
    const Tensor& t) {
  void (*result)(CTYPE_COMPUTE, void*) = nullptr;
  ET_SWITCH_TWO_TYPES(
      Bool,
      Byte,
      t.scalar_type(),
      context,
      op_name,
      TENSOR_CTYPE,
      [&]() -> void {
        result = internal::convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>;
      });
  return result;
}

template <typename CTYPE_COMPUTE, const char* op_name>
store_compute_to_tensor_fn<CTYPE_COMPUTE>
get_store_compute_to_tensor_fn_same_as_compute(
    KernelRuntimeContext& context,
    const Tensor& t) {
  void (*result)(CTYPE_COMPUTE, void*) = nullptr;
  constexpr auto common_scalar_type = CppTypeToScalarType<CTYPE_COMPUTE>::value;
  if (t.scalar_type() != common_scalar_type) {
    context.fail(torch::executor::Error::InvalidArgument);
    ET_LOG(
        Error,
        "Unhandled dtype %s for %s",
        ::executorch::runtime::toString(t.scalar_type()),
        op_name);
  } else {
    result = internal::convert_and_store<CTYPE_COMPUTE, CTYPE_COMPUTE>;
  }
  return result;
}

template <
    typename CTYPE_COMPUTE,
    const char* op_name,
    std::enable_if_t<std::is_same_v<CTYPE_COMPUTE, float>, bool> = true>
store_compute_to_tensor_fn<CTYPE_COMPUTE>
get_store_compute_to_tensor_fn_same_as_common(
    KernelRuntimeContext& context,
    const Tensor& t) {
  void (*result)(CTYPE_COMPUTE, void*) = nullptr;
  ET_SWITCH_THREE_TYPES(
      Float,
      Half,
      BFloat16,
      t.scalar_type(),
      context,
      op_name,
      CTYPE,
      [&]() -> void {
        result = internal::convert_and_store<CTYPE, CTYPE_COMPUTE>;
      });
  return result;
}

template <
    typename CTYPE_COMPUTE,
    const char* op_name,
    std::enable_if_t<!std::is_same_v<CTYPE_COMPUTE, float>, bool> = true>
store_compute_to_tensor_fn<CTYPE_COMPUTE>
get_store_compute_to_tensor_fn_same_as_common(
    KernelRuntimeContext& context,
    const Tensor& t) {
  return get_store_compute_to_tensor_fn_same_as_compute<CTYPE_COMPUTE, op_name>(
      context, t);
}

} // namespace internal

enum class SupportedTensorDtypes {
  REALHBBF16,
  REALHBF16,
  FLOATHBF16,
  INTB,
  BOOL,
  BOOL_OR_BYTE,
  // DEPRECATED: not likely to be correct; use SAME_AS_COMMON.
  SAME_AS_COMPUTE,
  SAME_AS_COMMON,
};

namespace internal {

template <typename CTYPE_COMPUTE, const char* op_name>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_impl(
    KernelRuntimeContext& context,
    const Tensor& t,
    SupportedTensorDtypes dtypes) {
  switch (dtypes) {
    case SupportedTensorDtypes::REALHBBF16:
      return get_load_to_compute_fn_realhbbf16<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::REALHBF16:
      return get_load_to_compute_fn_realhbf16<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::FLOATHBF16:
      return get_load_to_compute_fn_realhbf16<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::INTB:
      return get_load_to_compute_fn_intb<CTYPE_COMPUTE, op_name>(context, t);
    case SupportedTensorDtypes::BOOL:
      return get_load_to_compute_fn_bool<CTYPE_COMPUTE, op_name>(context, t);
    case SupportedTensorDtypes::BOOL_OR_BYTE:
      return get_load_to_compute_fn_bool_or_byte<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::SAME_AS_COMPUTE:
      return get_load_to_compute_fn_same_as_compute<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::SAME_AS_COMMON:
      return get_load_to_compute_fn_same_as_common<CTYPE_COMPUTE, op_name>(
          context, t);
  }
  ET_CHECK(false);
  return nullptr;
}

// NOTE: applying the #ifdef EXECUTORCH_SELECTIVE_BUILD_DTYPE
// technique used for get_load_to_compute_fn in this path was a size
// regression rather than an improvement. Haven't fully investigated
// why; just be aware when trying to improve size further.
template <typename CTYPE_COMPUTE, const char* op_name>
store_compute_to_tensor_fn<CTYPE_COMPUTE> get_store_compute_to_tensor_fn(
    KernelRuntimeContext& context,
    const Tensor& t,
    SupportedTensorDtypes dtypes) {
  switch (dtypes) {
    case SupportedTensorDtypes::REALHBBF16:
      return get_store_compute_to_tensor_fn_realhbbf16<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::REALHBF16:
      return get_store_compute_to_tensor_fn_realhbf16<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::FLOATHBF16:
      return get_store_compute_to_tensor_fn_floathbf16<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::INTB:
      return get_store_compute_to_tensor_fn_intb<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::BOOL:
      return get_store_compute_to_tensor_fn_bool<CTYPE_COMPUTE, op_name>(
          context, t);
    case SupportedTensorDtypes::BOOL_OR_BYTE:
      return get_store_compute_to_tensor_fn_bool_or_byte<
          CTYPE_COMPUTE,
          op_name>(context, t);
    case SupportedTensorDtypes::SAME_AS_COMPUTE:
      return get_store_compute_to_tensor_fn_same_as_compute<
          CTYPE_COMPUTE,
          op_name>(context, t);
    case SupportedTensorDtypes::SAME_AS_COMMON: {
      return get_store_compute_to_tensor_fn_same_as_common<
          CTYPE_COMPUTE,
          op_name>(context, t);
    }
  }
  ET_CHECK(false);
  return nullptr;
}

#ifndef EXECUTORCH_SELECTIVE_BUILD_DTYPE
inline constexpr const char kGenericElementwiseOpName[] =
    "generic_elementwise_op";
#endif // EXECUTORCH_SELECTIVE_BUILD_DTYPE

template <typename CTYPE_COMPUTE, const char* op_name>
load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn(
    KernelRuntimeContext& context,
    const Tensor& t,
    SupportedTensorDtypes dtypes) {
  // NOTE: Selective build relies on the operator name being passed
  // here. When it's *not* active, using the same operator name
  // everywhere saves on size because we don't require a new template
  // instantiation for every operator.
  return get_load_to_compute_fn_impl<
      CTYPE_COMPUTE,
#ifdef EXECUTORCH_SELECTIVE_BUILD_DTYPE
      op_name
#else // EXECUTORCH_SELECTIVE_BUILD_DTYPE
      kGenericElementwiseOpName
#endif // EXECUTORCH_SELECTIVE_BUILD_DTYPE
      >(context, t, dtypes);
}

bool check_tensor_dtype(
    const Tensor t,
    SupportedTensorDtypes dtypes,
    const ScalarType compute_type);

/// Return the one output type we are willing to emit specialized code
/// to handle, given a compute type of CTYPE_COMPUTE and supported
/// output types of out_dtypes.
template <typename CTYPE_COMPUTE>
inline constexpr ScalarType specialized_output_scalar_type(
    SupportedTensorDtypes out_dtypes) {
  switch (out_dtypes) {
    case SupportedTensorDtypes::BOOL:
      return ScalarType::Bool;
    case SupportedTensorDtypes::BOOL_OR_BYTE:
      return ScalarType::Bool;
    case SupportedTensorDtypes::REALHBBF16:
    case SupportedTensorDtypes::REALHBF16:
    case SupportedTensorDtypes::FLOATHBF16:
    case SupportedTensorDtypes::INTB:
    case SupportedTensorDtypes::SAME_AS_COMPUTE:
    case SupportedTensorDtypes::SAME_AS_COMMON:
      return CppTypeToScalarType<CTYPE_COMPUTE>::value;
  }
}

} // namespace internal
} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
