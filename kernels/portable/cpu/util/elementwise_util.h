/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace torch {
namespace executor {
namespace native {
namespace utils {

/*
 * Convert Scalar to C++ type
 */

template <typename T>
T scalar_to(const Scalar& s) {
  if (s.isBoolean()) {
    return static_cast<T>(s.to<bool>());
  } else if (s.isFloatingPoint()) {
    return static_cast<T>(s.to<double>());
  } else {
    return static_cast<T>(s.to<int64_t>());
  }
}

template <>
inline double scalar_to<double>(const Scalar& s) {
  return s.isFloatingPoint() ? s.to<double>()
                             : static_cast<double>(s.to<int64_t>());
}

template <>
inline int64_t scalar_to<int64_t>(const Scalar& s) {
  return s.isFloatingPoint() ? static_cast<int64_t>(s.to<double>())
                             : s.to<int64_t>();
}

template <typename CTYPE_COMMON, const char* op_name, typename Op>
inline void apply_unitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes) {
  constexpr auto compute_type = CppTypeToScalarType<CTYPE_COMMON>::value;

  ET_KERNEL_CHECK(
      ctx,
      (internal::check_tensor_dtype(a, a_dtypes, compute_type) &&
       internal::check_tensor_dtype(out, out_dtypes, compute_type)),
      InvalidArgument, );

  const auto load_a_to_common =
      internal::get_load_to_common_fn<CTYPE_COMMON, op_name>(a, a_dtypes);
  const auto store_common_to_out =
      internal::get_store_common_to_tensor_fn<CTYPE_COMMON, op_name>(
          out, out_dtypes);
  const char* const data_a = reinterpret_cast<const char*>(a.const_data_ptr());
  const auto a_element_size = a.element_size();
  const auto out_element_size = out.element_size();
  char* const data_out = reinterpret_cast<char*>(out.mutable_data_ptr());

  auto out_numel = out.numel();
  for (size_t i = 0; i < out_numel; ++i) {
    auto result = compute_fun(load_a_to_common(&data_a[i * a_element_size]));
    store_common_to_out(result, &data_out[i * out_element_size]);
  }
}

/**
 * Useful for bi-tensor elementwise operators. For each element of the inputs,
 * perform a computation and write to the corresponding element of the output.
 * Tensor broadcasting is applied wherever it is required.
 */
template <typename CTYPE_COMMON, const char* op_name, typename Op>
inline void apply_bitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& b,
    SupportedTensorDtypes b_dtypes,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes) {
  constexpr auto compute_type = CppTypeToScalarType<CTYPE_COMMON>::value;

  ET_KERNEL_CHECK(
      ctx,
      (internal::check_tensor_dtype(a, a_dtypes, compute_type) &&
       internal::check_tensor_dtype(b, b_dtypes, compute_type) &&
       internal::check_tensor_dtype(out, out_dtypes, compute_type)),
      InvalidArgument, );

  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool any_is_broadcasted = (a_is_broadcasted || b_is_broadcasted);

  const auto load_a_to_common =
      internal::get_load_to_common_fn<CTYPE_COMMON, op_name>(a, a_dtypes);
  const auto load_b_to_common =
      internal::get_load_to_common_fn<CTYPE_COMMON, op_name>(b, b_dtypes);
  const auto store_common_to_out =
      internal::get_store_common_to_tensor_fn<CTYPE_COMMON, op_name>(
          out, out_dtypes);
  const char* const data_a = reinterpret_cast<const char*>(a.const_data_ptr());
  const char* const data_b = reinterpret_cast<const char*>(b.const_data_ptr());
  const auto a_element_size = a.element_size();
  const auto b_element_size = b.element_size();
  const auto out_element_size = out.element_size();
  char* const data_out = reinterpret_cast<char*>(out.mutable_data_ptr());

  auto out_numel = out.numel();
  for (size_t i = 0; i < out_numel; ++i) {
    size_t a_linear_index = i;
    size_t b_linear_index = i;

    if (any_is_broadcasted) {
      size_t out_indexes[kTensorDimensionLimit];
      delinearize_index(i, out, out_indexes, kTensorDimensionLimit);

      if (a_is_broadcasted) {
        a_linear_index = linearize_access_indexes(out_indexes, out.dim(), a);
      }
      if (b_is_broadcasted) {
        b_linear_index = linearize_access_indexes(out_indexes, out.dim(), b);
      }
    }

    auto result = compute_fun(
        load_a_to_common(&data_a[a_linear_index * a_element_size]),
        load_b_to_common(&data_b[b_linear_index * b_element_size]));
    store_common_to_out(result, &data_out[i * out_element_size]);
  }
}

/**
 * Useful for tri-tensor elementwise operators. For each element of the
 * inputs, perform a computation and write to the corresponding element of the
 * output. Tensor broadcasting is applied wherever it is required.
 *
 * In order to mitigate build time cost (straightforwardly |CTYPE_A| *
 * |CTYPE_B| * |CTYPE_C| * |CTYPE_OUT|), all arguments to compute_fun
 * are passed as CTYPE_COMMON.
 *
 * Each tensor's supported dtypes set must be provided. The tensor
 * will be checked to ensure that its dtype falls into that set.
 *
 * op_name is used to support dtype selective build, as with the
 * ET_SWITCH family of macros. Note: because of C++17 quirks, you
 * can't pass a string literal for op_name. Instead, you should do the
 * following:
 *
 * static constexpr const char op_name[] = "my_op";
 * apply_ternary_elementwise_fn<CTYPE_COMMON, op_name>.
 */
template <typename CTYPE_COMMON, const char* op_name, typename Op>
inline void apply_tritensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& b,
    SupportedTensorDtypes b_dtypes,
    const Tensor& c,
    SupportedTensorDtypes c_dtypes,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes) {
  constexpr auto compute_type = CppTypeToScalarType<CTYPE_COMMON>::value;

  ET_KERNEL_CHECK(
      ctx,
      (internal::check_tensor_dtype(a, a_dtypes, compute_type) &&
       internal::check_tensor_dtype(b, b_dtypes, compute_type) &&
       internal::check_tensor_dtype(c, c_dtypes, compute_type) &&
       internal::check_tensor_dtype(out, out_dtypes, compute_type)),
      InvalidArgument, );

  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool c_is_broadcasted = !out.sizes().equals(c.sizes());
  const bool any_is_broadcasted =
      (a_is_broadcasted || b_is_broadcasted || c_is_broadcasted);

  const auto load_a_to_common =
      internal::get_load_to_common_fn<CTYPE_COMMON, op_name>(a, a_dtypes);
  const auto load_b_to_common =
      internal::get_load_to_common_fn<CTYPE_COMMON, op_name>(b, b_dtypes);
  const auto load_c_to_common =
      internal::get_load_to_common_fn<CTYPE_COMMON, op_name>(c, c_dtypes);
  const auto store_common_to_out =
      internal::get_store_common_to_tensor_fn<CTYPE_COMMON, op_name>(
          out, out_dtypes);
  const char* const data_a = reinterpret_cast<const char*>(a.const_data_ptr());
  const char* const data_b = reinterpret_cast<const char*>(b.const_data_ptr());
  const char* const data_c = reinterpret_cast<const char*>(c.const_data_ptr());
  const auto a_element_size = a.element_size();
  const auto b_element_size = b.element_size();
  const auto c_element_size = c.element_size();
  const auto out_element_size = out.element_size();
  char* const data_out = reinterpret_cast<char*>(out.mutable_data_ptr());

  auto out_numel = out.numel();
  for (size_t i = 0; i < out_numel; ++i) {
    size_t a_linear_index = i;
    size_t b_linear_index = i;
    size_t c_linear_index = i;

    if (any_is_broadcasted) {
      size_t out_indexes[kTensorDimensionLimit];
      delinearize_index(i, out, out_indexes, kTensorDimensionLimit);

      if (a_is_broadcasted) {
        a_linear_index = linearize_access_indexes(out_indexes, out.dim(), a);
      }
      if (b_is_broadcasted) {
        b_linear_index = linearize_access_indexes(out_indexes, out.dim(), b);
      }
      if (c_is_broadcasted) {
        c_linear_index = linearize_access_indexes(out_indexes, out.dim(), c);
      }
    }

    auto result = compute_fun(
        load_a_to_common(&data_a[a_linear_index * a_element_size]),
        load_b_to_common(&data_b[b_linear_index * b_element_size]),
        load_c_to_common(&data_c[c_linear_index * c_element_size]));
    store_common_to_out(result, &data_out[i * out_element_size]);
  }
}

inline ScalarType get_compute_type(ScalarType& common_type) {
  ScalarType compute_type = common_type;
  if (common_type == ScalarType::Half || common_type == ScalarType::BFloat16) {
    compute_type = ScalarType::Float;
  }
  return compute_type;
}

} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
