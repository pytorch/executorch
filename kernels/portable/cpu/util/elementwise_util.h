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

#ifdef ET_USE_PYTORCH_HEADERS
#include <ATen/cpu/vec/vec.h>
#endif // ET_USE_PYTORCH_HEADERS

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
template <typename Ignore, typename T>
using ignore_first_yield_second = T;

template <typename CTYPE_COMMON, typename Op, typename... Args>
using op_call_result =
    std::invoke_result_t<Op, ignore_first_yield_second<Args, CTYPE_COMMON>...>;

#ifdef ET_USE_PYTORCH_HEADERS
template <typename T>
struct is_vectorized : public std::false_type {};

template <typename T>
struct is_vectorized<at::vec::Vectorized<T>> : public std::true_type {};

// TODO: can_use_vectorized and can_use_vectorized_impl are a failed
// attempt to use SFINAE to detect whether our generic lambda argument
// with deduced return type would compile if it was passed
// Vectorized<CTYPE_COMMON> instead of CTYPE_COMMON. SFINAE does not
// work that way (see
// e.g. https://stackoverflow.com/questions/53344484/hard-error-when-using-stdinvoke-result-t-with-a-generic-lambda,
// https://stackoverflow.com/questions/31368601/how-to-detect-if-a-generic-lambda-is-uncompilable-in-c-14);
// if we really want to do it then we need to at least require that
// our lambdas actively participate in being SFINAE-friendly, as in
// https://stackoverflow.com/questions/76525790/detecting-if-a-generic-lambda-with-certain-arguments-is-invocable.
template <typename CTYPE_COMMON, typename Op, typename Enable=void, typename... Args>
struct can_use_vectorized_impl : std::false_type {};
template <typename CTYPE_COMMON, typename Op, typename... Args>
struct can_use_vectorized_impl<CTYPE_COMMON, Op, typename std::void_t<decltype(std::declval<std::invoke_result_t<
      Op,
                                                                               ignore_first_yield_second<Args, at::vec::Vectorized<CTYPE_COMMON>>...>>().store(std::declval<CTYPE_COMMON*>()))>, Args...> : public std::true_type {};//std::bool_constant<is_vectorized<std::invoke_result_t<Op,ignore_first_yield_second<Args, at::vec::Vectorized<CTYPE_COMMON>>...>>::value> {};

// Can I call a function of type Op with sizeof...(Args) arguments of type
// at::vec::Vectorized<CTYPE_COMMON>?
// This is not possible in C++17 as the code is currently set up; see TODO above.
template <typename CTYPE_COMMON, typename Op, typename...Args>
struct can_use_vectorized : public can_use_vectorized_impl<CTYPE_COMMON, Op, void, Args...> {};

#endif // ET_USE_PYTORCH_HEADERS

template <
    typename CTYPE_COMMON,
    typename CTYPE_OUT,
    typename Op,
    typename... Args>
inline void dtype_specialized_elementwise_fn_impl(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& out,
    Args... inputs) {
  static_assert(
      (std::is_same_v<Args, std::pair<const Tensor*, SupportedTensorDtypes>> &&
       ...));
  constexpr auto kNumInputs = sizeof...(inputs);
  // All inputs must be of type CTYPE_COMMON.
  ET_DCHECK(
      ((inputs.first->scalar_type() ==
        CppTypeToScalarType<CTYPE_COMMON>::value) &&
       ...));

  std::array<const CTYPE_COMMON*, kNumInputs> inputs_data_ptrs = {
      inputs.first->template const_data_ptr<CTYPE_COMMON>()...};

  CTYPE_OUT* const data_out = out.mutable_data_ptr<CTYPE_OUT>();

#ifdef ET_USE_PYTORCH_HEADERS
  if constexpr (can_use_vectorized<CTYPE_COMMON, Op, Args...>::value) {
    const bool any_is_broadcasted =
        !(torch::executor::internal::sizes_match_ignoring_leading_1s(
              inputs.first->sizes(), out.sizes()) &&
          ...);
    if (!any_is_broadcasted) {
      using Vec = at::vec::Vectorized<CTYPE_COMMON>;
      ::executorch::extension::parallel_for(
          0,
          out.numel(),
          ::executorch::extension::internal::GRAIN_SIZE,
          [&](const auto begin, const auto end) {
            const auto vectorized_begin =
                begin + (Vec::size() - begin % Vec::size()) % Vec::size();
            const auto vectorized_end = end - (end % Vec::size());
            // Scalar prologue.
            for (const auto idx : c10::irange(begin, vectorized_begin)) {
              std::array<CTYPE_COMMON, kNumInputs> loaded_inputs;
              for (const auto input_idx : c10::irange(kNumInputs)) {
                loaded_inputs[input_idx] = inputs_data_ptrs[input_idx][idx];
              }
              data_out[idx] = std::apply(compute_fun, loaded_inputs);
            }

            // Main vectorized loop.
            for (auto idx = vectorized_begin; idx < vectorized_end;
                 idx += Vec::size()) {
              std::array<Vec, kNumInputs> loaded_vec_inputs;
              for (const auto input_idx : c10::irange(kNumInputs)) {
                loaded_vec_inputs[input_idx] =
                    Vec::loadu(&inputs_data_ptrs[input_idx][idx]);
              }
              auto result_vec = std::apply(compute_fun, loaded_vec_inputs);
              result_vec.store(&data_out[idx]);
            }

            // Scalar epilogue.
            for (const auto idx : c10::irange(vectorized_end, end)) {
              std::array<CTYPE_COMMON, kNumInputs> loaded_inputs;
              for (const auto input_idx : c10::irange(kNumInputs)) {
                loaded_inputs[input_idx] = inputs_data_ptrs[input_idx][idx];
              }
              data_out[idx] = std::apply(compute_fun, loaded_inputs);
            }
          });
      return;
    }
  }
#endif

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
            loaded_inputs[idx] = inputs_data_ptrs[idx][indexes[idx + 1]];
          }
          data_out[indexes[0]] = std::apply(compute_fun, loaded_inputs);
        }
      });
}

template <typename CTYPE_COMMON, typename Op, typename... Args>
inline bool validate_elementwise_fn_inputs(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    Args... inputs) {
  static_assert(
      (std::is_same_v<Args, std::pair<const Tensor*, SupportedTensorDtypes>> &&
       ...));
  constexpr auto compute_type = CppTypeToScalarType<CTYPE_COMMON>::value;
  const auto check_input_dtype = [](auto input, auto compute_type) {
    return internal::check_tensor_dtype(
        *input.first, input.second, compute_type);
  };
  ET_KERNEL_CHECK(
      ctx,
      (check_input_dtype(inputs, compute_type) && ...) &&
          internal::check_tensor_dtype(out, out_dtypes, compute_type),
      InvalidArgument,
      false);

  return true;
}

template <
    typename CTYPE_COMMON,
    const char* op_name,
    typename Op,
    typename... Args>
inline void apply_elementwise_fn_generic_impl(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    Args... inputs) {
  constexpr auto kNumInputs = sizeof...(inputs);

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

  // NOTE: the result of compute_fun is not necessarily CTYPE_COMMON!
  // For example, consider the possibility that compute_fun is a
  // trigonometric function like acos, the common input type is bool,
  // and the output type is float -- we would truncate acos(0) ~= 1.67
  // to just 1. Conveniently, it costs us nothing at runtime to handle
  // this correctly.
  const auto store_compute_result_to_out =
      internal::get_store_common_to_tensor_fn<
          op_call_result<CTYPE_COMMON, Op, Args...>,
          op_name>(out, out_dtypes);
  char* const data_out = reinterpret_cast<char*>(out.mutable_data_ptr());
  const auto out_element_size = out.element_size();

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
          store_compute_result_to_out(
              result, &data_out[indexes[0] * out_element_size]);
        }
      });
}

template <
    typename CTYPE_COMMON,
    const char* op_name,
    typename Op,
    typename... Args>
inline void apply_elementwise_fn_runtime_out_dtypes(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    Args... inputs) {
  const bool inputs_valid = validate_elementwise_fn_inputs<CTYPE_COMMON>(
      compute_fun, ctx, out, out_dtypes, inputs...);
  if (!inputs_valid) {
    return;
  }

  apply_elementwise_fn_generic_impl<CTYPE_COMMON, op_name>(
      compute_fun, ctx, out, out_dtypes, inputs...);
}

template <
    typename CTYPE_COMMON,
    const char* op_name,
    SupportedTensorDtypes out_dtypes,
    typename Op,
    typename... Args>
inline void apply_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& out,
    Args... inputs) {
  const bool inputs_valid = validate_elementwise_fn_inputs<CTYPE_COMMON>(
      compute_fun, ctx, out, out_dtypes, inputs...);
  if (!inputs_valid) {
    return;
  }

  constexpr auto compute_type = CppTypeToScalarType<CTYPE_COMMON>::value;
  const bool all_inputs_compute_dtype =
      ((inputs.first->scalar_type() == compute_type) && ...);

  constexpr ScalarType out_specialized_scalar_type =
      specialized_output_scalar_type<CTYPE_COMMON>(out_dtypes);
  if (all_inputs_compute_dtype &&
      out.scalar_type() == out_specialized_scalar_type) {
    using CTYPE_OUT =
        typename ScalarTypeToCppType<out_specialized_scalar_type>::type;
    dtype_specialized_elementwise_fn_impl<CTYPE_COMMON, CTYPE_OUT>(
        compute_fun, ctx, out, inputs...);
    return;
  }

  apply_elementwise_fn_generic_impl<CTYPE_COMMON, op_name>(
      compute_fun, ctx, out, out_dtypes, inputs...);
}
} // namespace internal

/// DEPRECATED: prefer the variant with out_dtypes in the template argument.
template <typename CTYPE_COMMON, const char* op_name, typename Op>
inline void apply_unitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes) {
  internal::apply_elementwise_fn_runtime_out_dtypes<CTYPE_COMMON, op_name>(
      compute_fun, ctx, out, out_dtypes, std::make_pair(&a, a_dtypes));
}

template <
    typename CTYPE_COMMON,
    const char* op_name,
    SupportedTensorDtypes out_dtypes,
    typename Op>
inline void apply_unitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& out) {
  internal::apply_elementwise_fn<CTYPE_COMMON, op_name, out_dtypes>(
      compute_fun, ctx, out, std::make_pair(&a, a_dtypes));
}

/**
 * DEPRECATED: prefer the variant with out_dtypes in the template argument list.
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
  internal::apply_elementwise_fn_runtime_out_dtypes<CTYPE_COMMON, op_name>(
      compute_fun,
      ctx,
      out,
      out_dtypes,
      std::make_pair(&a, a_dtypes),
      std::make_pair(&b, b_dtypes));
}

/**
 * Useful for bi-tensor elementwise operators. For each element of the inputs,
 * perform a computation and write to the corresponding element of the output.
 * Tensor broadcasting is applied wherever it is required.
 */
template <
    typename CTYPE_COMMON,
    const char* op_name,
    SupportedTensorDtypes out_dtypes,
    typename Op>
inline void apply_bitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& b,
    SupportedTensorDtypes b_dtypes,
    const Tensor& out) {
  internal::apply_elementwise_fn<CTYPE_COMMON, op_name, out_dtypes>(
      compute_fun,
      ctx,
      out,
      std::make_pair(&a, a_dtypes),
      std::make_pair(&b, b_dtypes));
}

/**
 * DEPRECATED: prefer the variant with out_dtypes in the template argument list.
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
  internal::apply_elementwise_fn_runtime_out_dtypes<CTYPE_COMMON, op_name>(
      compute_fun,
      ctx,
      out,
      out_dtypes,
      std::make_pair(&a, a_dtypes),
      std::make_pair(&b, b_dtypes),
      std::make_pair(&c, c_dtypes));
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
template <
    typename CTYPE_COMMON,
    const char* op_name,
    SupportedTensorDtypes out_dtypes,
    typename Op>
inline void apply_tritensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& b,
    SupportedTensorDtypes b_dtypes,
    const Tensor& c,
    SupportedTensorDtypes c_dtypes,
    const Tensor& out) {
  internal::apply_elementwise_fn<CTYPE_COMMON, op_name, out_dtypes>(
      compute_fun,
      ctx,
      out,
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
