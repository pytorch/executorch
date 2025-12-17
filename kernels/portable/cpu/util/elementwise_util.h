/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/selective_build.h>
#include <executorch/kernels/portable/cpu/util/broadcast_indexes_range.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/kernels/portable/cpu/util/vectorized_math.h> // Make vectorization support easy for clients.
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

#if defined(ET_USE_PYTORCH_HEADERS) && ET_USE_PYTORCH_HEADERS
#include <ATen/cpu/vec/vec.h>
#endif // ET_USE_PYTORCH_HEADERS

#include <array>
#include <utility>

namespace torch {
namespace executor {
namespace native {
namespace utils {
namespace internal {
/**
 * Causes these utility functions to make sure to respect Tensor
 * strides; normally, this is not strictly necessary because ExecuTorch
 * Tensors are contiguous.
 */
struct SupportNoncontiguousInputTensors {
  explicit SupportNoncontiguousInputTensors() = default;
};

template <typename Ignore, typename T>
using ignore_first_yield_second = T;

#if defined(ET_USE_PYTORCH_HEADERS) && ET_USE_PYTORCH_HEADERS
// Can I call a function of type Op with sizeof...(Args) arguments of type
// at::vec::Vectorized<CTYPE_COMPUTE>?
//
// See [NOTE: Generic lambdas] below for requirements on Op.
template <typename CTYPE_COMPUTE, typename Op, typename... Args>
constexpr bool can_use_vectorized() {
  using Vec = at::vec::Vectorized<CTYPE_COMPUTE>;
  // NOTE: if we start building optimized kernels on platforms that
  // ATen Vectorized doesn't support well, we will want to add a way
  // to check that Vectorized actually does something on our target
  // platform. For now, I see no concrete need for that.
  if constexpr (std::is_invocable_v<
                    Op,
                    ignore_first_yield_second<Args, Vec>...>) {
    // For bool, we will get a false positive if we rely on only the
    // is_invocable_v check above because at::vec::Vectorized is
    // implicitly convertible to a pointer, which makes it implicitly
    // convertible to bool (which was 15 minutes of fun to debug). Also
    // just seems like good hygiene to make sure we get the Vectorized
    // we're expecting.
    return std::is_same_v<
        std::invoke_result_t<Op, ignore_first_yield_second<Args, Vec>...>,
        Vec>;
  }
  return false;
}
#endif // ET_USE_PYTORCH_HEADERS

template <
    typename CTYPE_COMPUTE,
    typename CTYPE_OUT,
    bool support_noncontiguous_tensors,
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
  static constexpr auto kNumInputs = sizeof...(inputs);
  // All inputs must be of type CTYPE_COMPUTE.
  ET_DCHECK(
      ((inputs.first->scalar_type() ==
        CppTypeToScalarType<CTYPE_COMPUTE>::value) &&
       ...));

#if defined(ET_USE_PYTORCH_HEADERS) && ET_USE_PYTORCH_HEADERS
  if constexpr (can_use_vectorized<CTYPE_COMPUTE, Op, Args...>()) {
    const bool any_is_broadcasted =
        !(torch::executor::internal::sizes_match_ignoring_leading_1s(
              inputs.first->sizes(), out.sizes()) &&
          ...);
    if (!any_is_broadcasted) {
      using Vec = at::vec::Vectorized<CTYPE_COMPUTE>;
      ::executorch::extension::parallel_for(
          0,
          out.numel(),
          ::executorch::extension::internal::GRAIN_SIZE,
          [&](const auto begin, const auto end) {
            std::array<const CTYPE_COMPUTE*, kNumInputs> inputs_data_ptrs = {
                inputs.first->template const_data_ptr<CTYPE_COMPUTE>()...};

            CTYPE_OUT* const data_out = out.mutable_data_ptr<CTYPE_OUT>();

            const auto vectorized_begin =
                begin + (Vec::size() - begin % Vec::size()) % Vec::size();
            const auto vectorized_end = end - (end % Vec::size());
            // Scalar prologue.
            for (const auto idx : c10::irange(begin, vectorized_begin)) {
          // In debug mode, always use Vectorized so that even
          // small-sized tests will test whether using Vectorized broke our
          // lambda.
#ifndef NDEBUG
              std::array<Vec, kNumInputs> loaded_inputs{};
#else // NDEBUG
              std::array<CTYPE_COMPUTE, kNumInputs> loaded_inputs{};
#endif // NDEBUG
              for (const auto input_idx : c10::irange(kNumInputs)) {
                loaded_inputs[input_idx] = inputs_data_ptrs[input_idx][idx];
              }
#ifndef NDEBUG
              std::apply(compute_fun, loaded_inputs).store(&data_out[idx], 1);
#else // NDEBUG
              data_out[idx] = std::apply(compute_fun, loaded_inputs);
#endif // NDEBUG
            }

            // Main vectorized loop.
            for (auto idx = vectorized_begin; idx < vectorized_end;
                 idx += Vec::size()) {
              std::array<Vec, kNumInputs> loaded_vec_inputs{};
              for (const auto input_idx : c10::irange(kNumInputs)) {
                loaded_vec_inputs[input_idx] =
                    Vec::loadu(&inputs_data_ptrs[input_idx][idx]);
              }
              auto result_vec = std::apply(compute_fun, loaded_vec_inputs);
              result_vec.store(&data_out[idx]);
            }

            // Scalar epilogue.
            for (const auto idx : c10::irange(vectorized_end, end)) {
#ifndef NDEBUG
              std::array<Vec, kNumInputs> loaded_inputs{};
#else // NDEBUG
              std::array<CTYPE_COMPUTE, kNumInputs> loaded_inputs{};
#endif // NDEBUG
              for (const auto input_idx : c10::irange(kNumInputs)) {
                loaded_inputs[input_idx] = inputs_data_ptrs[input_idx][idx];
              }
#ifndef NDEBUG
              std::apply(compute_fun, loaded_inputs).store(&data_out[idx], 1);
#else // NDEBUG
              data_out[idx] = std::apply(compute_fun, loaded_inputs);
#endif // NDEBUG
            }
          });
      return;
    }
  }
#endif // ET_USE_PYTORCH_HEADERS

  ::executorch::extension::parallel_for(
      0,
      out.numel(),
      ::executorch::extension::internal::GRAIN_SIZE,
      [&](const auto begin, const auto end) {
        std::array<const CTYPE_COMPUTE*, kNumInputs> inputs_data_ptrs = {
            inputs.first->template const_data_ptr<CTYPE_COMPUTE>()...};

        CTYPE_OUT* const data_out = out.mutable_data_ptr<CTYPE_OUT>();

        const auto range =
            BroadcastIndexesRange<kNumInputs, support_noncontiguous_tensors>(
                out, (*inputs.first)...);
        auto begin_it = range.begin();
        begin_it += begin;
        for (; (*begin_it)[0] < end; ++begin_it) {
          const auto& indexes = *begin_it;
          std::array<CTYPE_COMPUTE, kNumInputs> loaded_inputs{};
          for (const auto idx : c10::irange(kNumInputs)) {
            loaded_inputs[idx] = inputs_data_ptrs[idx][indexes[idx + 1]];
          }
          data_out[indexes[0]] = std::apply(compute_fun, loaded_inputs);
        }
      });
}

template <typename CTYPE_COMPUTE, typename... Args>
inline bool validate_elementwise_fn_inputs(
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    Args... inputs) {
  static_assert(
      (std::is_same_v<Args, std::pair<const Tensor*, SupportedTensorDtypes>> &&
       ...));
  constexpr auto compute_type = CppTypeToScalarType<CTYPE_COMPUTE>::value;
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
    typename CTYPE_COMPUTE,
    const char* op_name,
    bool support_noncontiguous_tensors,
    typename Op,
    typename... Args>
inline void apply_elementwise_fn_generic_impl(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    Args... inputs) {
  static constexpr auto kNumInputs = sizeof...(inputs);

  struct InputInfo {
    load_to_compute_fn<CTYPE_COMPUTE> load_to_compute;
    const char* data_ptr;
    ssize_t element_size;
  };
  std::array<InputInfo, kNumInputs> inputs_info = {(InputInfo{
      internal::get_load_to_compute_fn<CTYPE_COMPUTE, op_name>(
          ctx, *inputs.first, inputs.second),
      reinterpret_cast<const char*>(inputs.first->const_data_ptr()),
      inputs.first->element_size(),
  })...};

  const auto store_compute_to_out =
      internal::get_store_compute_to_tensor_fn<CTYPE_COMPUTE, op_name>(
          ctx, out, out_dtypes);
  char* const data_out = reinterpret_cast<char*>(out.mutable_data_ptr());
  const auto out_element_size = out.element_size();

  ::executorch::extension::parallel_for(
      0,
      out.numel(),
      ::executorch::extension::internal::GRAIN_SIZE,
      [&](const auto begin, const auto end) {
        const auto range =
            BroadcastIndexesRange<kNumInputs, support_noncontiguous_tensors>(
                out, (*inputs.first)...);
        auto begin_it = range.begin();
        begin_it += begin;
        for (; (*begin_it)[0] < end; ++begin_it) {
          const auto& indexes = *begin_it;
          std::array<CTYPE_COMPUTE, kNumInputs> loaded_inputs{};
          for (const auto idx : c10::irange(kNumInputs)) {
            const auto& input_info = inputs_info[idx];
            loaded_inputs[idx] = input_info.load_to_compute(
                &input_info
                     .data_ptr[indexes[idx + 1] * input_info.element_size]);
          }
          auto result = std::apply(compute_fun, loaded_inputs);
          store_compute_to_out(
              result, &data_out[indexes[0] * out_element_size]);
        }
      });
}

template <
    typename CTYPE_COMPUTE,
    const char* op_name,
    typename Op,
    typename... Args>
inline void apply_elementwise_fn_runtime_out_dtypes(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes,
    Args... inputs) {
  const bool inputs_valid = validate_elementwise_fn_inputs<CTYPE_COMPUTE>(
      ctx, out, out_dtypes, inputs...);
  if (!inputs_valid) {
    return;
  }

  apply_elementwise_fn_generic_impl<
      CTYPE_COMPUTE,
      op_name,
      /*support_noncontiguous_tensors*/ false>(
      compute_fun, ctx, out, out_dtypes, inputs...);
}

template <
    typename CTYPE_COMPUTE,
    const char* op_name,
    SupportedTensorDtypes out_dtypes,
    bool support_noncontiguous_tensors,
    typename Op,
    typename... Args>
inline void apply_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& out,
    Args... inputs) {
  const bool inputs_valid = validate_elementwise_fn_inputs<CTYPE_COMPUTE>(
      ctx, out, out_dtypes, inputs...);
  if (!inputs_valid) {
    return;
  }

  constexpr auto compute_type = CppTypeToScalarType<CTYPE_COMPUTE>::value;
  constexpr ScalarType out_specialized_scalar_type =
      specialized_output_scalar_type<CTYPE_COMPUTE>(out_dtypes);
  if constexpr (should_include_kernel_dtype(
                    op_name, out_specialized_scalar_type)) {
    const bool all_inputs_compute_dtype =
        ((inputs.first->scalar_type() == compute_type) && ...);

    if (all_inputs_compute_dtype &&
        out.scalar_type() == out_specialized_scalar_type) {
      using CTYPE_OUT =
          typename ScalarTypeToCppType<out_specialized_scalar_type>::type;
      dtype_specialized_elementwise_fn_impl<
          CTYPE_COMPUTE,
          CTYPE_OUT,
          support_noncontiguous_tensors>(compute_fun, ctx, out, inputs...);
      return;
    }
  }

  apply_elementwise_fn_generic_impl<
      CTYPE_COMPUTE,
      op_name,
      support_noncontiguous_tensors>(
      compute_fun, ctx, out, out_dtypes, inputs...);
}

/// DEPRECATED: prefer the variant with out_dtypes in the template argument.
template <typename CTYPE_COMPUTE, const char* op_name, typename Op>
inline void apply_unitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes) {
  internal::apply_elementwise_fn_runtime_out_dtypes<CTYPE_COMPUTE, op_name>(
      compute_fun, ctx, out, out_dtypes, std::make_pair(&a, a_dtypes));
}

/**
 * Useful for unary elementwise operators. For each element of the
 * input, call Op and write to the corresponding element of the
 * output. Tensor broadcasting is applied wherever it is required.
 *
 * [NOTE: Generic lambdas]: If Op is a *generic* lambda (i.e., one with `auto`
 * parameters; normal lambdas are fine), it must fulfill one of the
 * following conditions. Either:
 * 1) It must in fact compile when passed at::vec::Vectorized<CTYPE_COMPUTE>, or
 * 2) It must be actively SFINAE-friendly, as per the C++17 examples in
 * https://stackoverflow.com/questions/76525790/detecting-if-a-generic-lambda-with-certain-arguments-is-invocable
 * .
 */
template <
    typename CTYPE_COMPUTE,
    const char* op_name,
    SupportedTensorDtypes out_dtypes,
    typename Op>
inline void apply_unitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& out) {
  internal::apply_elementwise_fn<
      CTYPE_COMPUTE,
      op_name,
      out_dtypes,
      /*support_noncontiguous_tensors*/ false>(
      compute_fun, ctx, out, std::make_pair(&a, a_dtypes));
}

template <
    typename CTYPE_COMPUTE,
    const char* op_name,
    SupportedTensorDtypes out_dtypes,
    typename Op>
inline void apply_unitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& out,
    SupportNoncontiguousInputTensors) {
  internal::apply_elementwise_fn<
      CTYPE_COMPUTE,
      op_name,
      out_dtypes,
      /*support_noncontiguous_tensors*/ true>(
      compute_fun, ctx, out, std::make_pair(&a, a_dtypes));
}

/**
 * DEPRECATED: prefer the variant with out_dtypes in the template argument list.
 */
template <typename CTYPE_COMPUTE, const char* op_name, typename Op>
inline void apply_bitensor_elementwise_fn(
    const Op& compute_fun,
    KernelRuntimeContext& ctx,
    const Tensor& a,
    SupportedTensorDtypes a_dtypes,
    const Tensor& b,
    SupportedTensorDtypes b_dtypes,
    const Tensor& out,
    SupportedTensorDtypes out_dtypes) {
  internal::apply_elementwise_fn_runtime_out_dtypes<CTYPE_COMPUTE, op_name>(
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
 * See [NOTE: Generic lambdas] if you want to pass a generic lambda for
 * compute_fun.
 */
template <
    typename CTYPE_COMPUTE,
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
  internal::apply_elementwise_fn<
      CTYPE_COMPUTE,
      op_name,
      out_dtypes,
      /*support_noncontiguous_tensors*/ false>(
      compute_fun,
      ctx,
      out,
      std::make_pair(&a, a_dtypes),
      std::make_pair(&b, b_dtypes));
}

template <
    typename CTYPE_COMPUTE,
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
    const Tensor& out,
    SupportNoncontiguousInputTensors) {
  internal::apply_elementwise_fn<
      CTYPE_COMPUTE,
      op_name,
      out_dtypes,
      /*support_noncontiguous_tensors*/ true>(
      compute_fun,
      ctx,
      out,
      std::make_pair(&a, a_dtypes),
      std::make_pair(&b, b_dtypes));
}

/**
 * DEPRECATED: prefer the variant with out_dtypes in the template argument list.
 */
template <typename CTYPE_COMPUTE, const char* op_name, typename Op>
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
  internal::apply_elementwise_fn_runtime_out_dtypes<CTYPE_COMPUTE, op_name>(
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
 * are passed as CTYPE_COMPUTE.
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
 * apply_ternary_elementwise_fn<CTYPE_COMPUTE, op_name>.
 *
 * See [NOTE: Generic lambdas] if you want to pass a generic lambda for
 * compute_fun.
 */
template <
    typename CTYPE_COMPUTE,
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
  internal::apply_elementwise_fn<
      CTYPE_COMPUTE,
      op_name,
      out_dtypes,
      /*support_noncontiguous_tensors*/ false>(
      compute_fun,
      ctx,
      out,
      std::make_pair(&a, a_dtypes),
      std::make_pair(&b, b_dtypes),
      std::make_pair(&c, c_dtypes));
}

template <
    typename CTYPE_COMPUTE,
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
    const Tensor& out,
    SupportNoncontiguousInputTensors) {
  internal::apply_elementwise_fn<
      CTYPE_COMPUTE,
      op_name,
      out_dtypes,
      /*support_noncontiguous_tensors*/ true>(
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
} // namespace internal

// DEPRECATED: these APIs should not have been stabilized for external
// use as they are undergoing active development.
using internal::apply_bitensor_elementwise_fn;
using internal::apply_tritensor_elementwise_fn;
using internal::apply_unitensor_elementwise_fn;
using internal::get_compute_type;

} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
