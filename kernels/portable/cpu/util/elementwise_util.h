/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/broadcast_indexes_range.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

#include <array>
#include <utility>

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

namespace internal {
template <
    typename CTYPE_COMMON,
    const char* op_name,
    typename Op,
    typename... Args>
inline void apply_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    Args... inputs) {
  static_assert(
      (std::is_same_v<Args, std::pair<const Tensor*, SupportedTensorDtypes>> &&
       ...));
  constexpr auto kNumInputs = sizeof...(inputs);
  constexpr auto compute_type = CppTypeToScalarType<CTYPE_COMMON>::value;
  const auto check_input_dtype = [](auto input, auto compute_type) {
    return internal::check_tensor_dtype(
        *input.first, input.second, compute_type);
  };
  ET_KERNEL_CHECK(
      ctx,
      (check_input_dtype(inputs, compute_type) && ...) &&
          internal::check_tensor_dtype(out, out_dtypes, compute_type),
      InvalidArgument, );

  bool any_is_broadcasted = false;
  if constexpr (kNumInputs > 1) {
    any_is_broadcasted = (!out.sizes().equals(inputs.first->sizes()) || ...);
  }

  struct InputInfo {
    load_to_common_fn<CTYPE_COMMON> load_to_common;
    const char* data_ptr;
    ssize_t element_size;
  };
  std::array<InputInfo, kNumInputs> inputs_info = {(InputInfo{
      internal::get_load_to_common_fn<CTYPE_COMMON, op_name>(
          *inputs.first, inputs.second),
      reinterpret_cast<const char*>(inputs.first->const_data_ptr()),
      inputs.first->element_size(),
  })...};

  const auto store_common_to_out =
      internal::get_store_common_to_tensor_fn<CTYPE_COMMON, op_name>(
          out, out_dtypes);
  char* const data_out = reinterpret_cast<char*>(out.mutable_data_ptr());
  const auto out_element_size = out.element_size();

  if (any_is_broadcasted) {
    ::executorch::extension::parallel_for(
        0,
        out.numel(),
        ::executorch::extension::internal::GRAIN_SIZE,
        [&](const auto begin, const auto end) {
          const auto range =
              BroadcastIndexesRange<kNumInputs>(out, (*inputs.first)...);
          auto begin_it = range.begin();
          begin_it += begin;
          for (; (*begin_it)[0] < end; ++begin_it) {
            const auto& indexes = *begin_it;
            std::array<CTYPE_COMMON, kNumInputs> loaded_inputs;
            for (const auto idx : c10::irange(kNumInputs)) {
              const auto& input_info = inputs_info[idx];
              loaded_inputs[idx] = input_info.load_to_common(
                  &input_info
                       .data_ptr[indexes[idx + 1] * input_info.element_size]);
            }
            auto result = std::apply(compute_fun, loaded_inputs);
            store_common_to_out(
                result, &data_out[indexes[0] * out_element_size]);
          }
        });
  } else {
    ::executorch::extension::parallel_for(
        0,
        out.numel(),
        ::executorch::extension::internal::GRAIN_SIZE,
        [&](const auto begin, const auto end) {
          for (const auto i : c10::irange(begin, end)) {
            std::array<CTYPE_COMMON, kNumInputs> loaded_inputs;
            for (const auto idx : c10::irange(kNumInputs)) {
              const auto& input_info = inputs_info[idx];
              loaded_inputs[idx] = input_info.load_to_common(
                  &input_info.data_ptr[i * input_info.element_size]);
            }
            auto result = std::apply(compute_fun, loaded_inputs);
            store_common_to_out(result, &data_out[i * out_element_size]);
          }
        });
  }
}
} // namespace internal

template <typename CTYPE_COMMON, const char* op_name, typename Op>
inline void apply_unitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes) {
  internal::apply_elementwise_fn<CTYPE_COMMON, op_name>(
      compute_fun, ctx, out, out_dtypes, std::make_pair(&a, a_dtypes));
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
  internal::apply_elementwise_fn<CTYPE_COMMON, op_name>(
      compute_fun,
      ctx,
      out,
      out_dtypes,
      std::make_pair(&a, a_dtypes),
      std::make_pair(&b, b_dtypes));
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
  internal::apply_elementwise_fn<CTYPE_COMMON, op_name>(
      compute_fun,
      ctx,
      out,
      out_dtypes,
      std::make_pair(&a, a_dtypes),
      std::make_pair(&b, b_dtypes),
      std::make_pair(&c, c_dtypes));
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
