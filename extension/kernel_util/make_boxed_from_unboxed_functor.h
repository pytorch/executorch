/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
/// \file extension/kernel_util/make_boxed_from_unboxed_functor.h
/// Defines a template that can be used to create a boxed version of an unboxed
/// functor.
/// Example usage:
/// ```
/// Tensor&
/// my_op(KernelRuntimeContext& ctx, const Tensor& self, const Tensor& other,
///       Tensor& out)
/// {
///   // ...
///   return out;
/// }
///
/// Kernel my_kernel = Kernel::make_boxed_kernel("my_ns::my_op",
///   EXECUTORCH_FN(my_op));
/// static auto res = register_kernels({my_kernel});
/// ```
/// Or simply:
/// ```
/// EXECUTORCH_LIBRARY(my_ns, "my_op", my_op);
/// ```
///
/// The trick here is to convert each EValue to inferred argument type. This
/// uses a lot of C++17 features.
//===----------------------------------------------------------------------===//

#pragma once
#if __cplusplus < 201703L
#error "This header requires C++17"
#endif

#include <executorch/extension/kernel_util/meta_programming.h>
#include <executorch/extension/kernel_util/type_list.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <typeinfo>

namespace executorch {
namespace runtime {
class KernelRuntimeContext; // Forward declaration
} // namespace runtime
} // namespace executorch

namespace executorch {
namespace extension {

// This extension has a lot of generic internal names like "size"; use a unique
// internal namespace to avoid conflicts with other extensions.
namespace kernel_util_internal {

template <class T>
struct decay_if_not_tensor final {
  using type = std::decay_t<T>;
};
template <>
struct decay_if_not_tensor<executorch::aten::Tensor&> final {
  using type = executorch::aten::Tensor&;
};
template <>
struct decay_if_not_tensor<const executorch::aten::Tensor&> final {
  using type = const executorch::aten::Tensor&;
};

template <class T>
struct evalue_to_arg final {
  static T call(executorch::runtime::EValue& v) {
    return std::move(v).to<T>();
  }
};

template <>
struct evalue_to_arg<executorch::aten::Tensor&> final {
  static executorch::aten::Tensor& call(executorch::runtime::EValue& v) {
    return v.toTensor();
  }
};

template <>
struct evalue_to_arg<const executorch::aten::Tensor&> final {
  static const executorch::aten::Tensor& call(executorch::runtime::EValue& v) {
    return v.toTensor();
  }
};

template <class T>
struct evalue_to_arg<executorch::aten::optional<T>> final {
  static executorch::aten::optional<T> call(executorch::runtime::EValue& v) {
    return v.toOptional<T>();
  }
};

template <class T>
struct evalue_to_arg<executorch::aten::ArrayRef<executorch::aten::optional<T>>>
    final {
  static executorch::aten::ArrayRef<executorch::aten::optional<T>> call(
      executorch::runtime::EValue& v) {
    return v.toListOptionalTensor();
  }
};

template <class Functor, size_t... evalue_arg_indices, typename... ArgTypes>
void call_functor_with_args_from_stack(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    executorch::runtime::EValue** stack,
    std::index_sequence<evalue_arg_indices...>,
    typelist<ArgTypes...>*) {
  (*Functor::func_ptr())(
      ctx,
      evalue_to_arg<typename decay_if_not_tensor<ArgTypes>::type>::call(
          *stack[evalue_arg_indices])...);
}

} // namespace kernel_util_internal

/**
 * WrapUnboxedIntoFunctor: Given a function pointer, wrap it into a functor that
 * takes EValues as input and returns void. The wrapped functor will unbox all
 * inputs and forward them to unboxed kernel.
 */
template <class FuncType>
struct WrapUnboxedIntoFunctor {
  static_assert(
      kernel_util_internal::is_compile_time_function_pointer<FuncType>::value,
      "Can't handle function other than EXECUTORCH_FN");
  using TrueType = typename FuncType::FuncType;
  using ReturnType = typename kernel_util_internal::infer_function_traits_t<
      TrueType>::return_type;
  using ArgsType = typename kernel_util_internal::infer_function_traits_t<
      TrueType>::parameter_types;
  // check if the first argument is KernelRuntimeContext, if so, remove it
  static constexpr bool first_arg_is_context = std::is_same<
      ::executorch::runtime::KernelRuntimeContext,
      std::remove_reference_t<
          kernel_util_internal::head_with_default_t<void, ArgsType>>>::value;
  using ContextRemovedArgsType = std::conditional_t<
      first_arg_is_context,
      kernel_util_internal::drop_if_nonempty_t<ArgsType, 1>,
      ArgsType>;

  static void call(
      ::executorch::runtime::KernelRuntimeContext& ctx,
      executorch::runtime::EValue** stack) {
    constexpr size_t num_inputs =
        kernel_util_internal::size<ContextRemovedArgsType>::value;
    return kernel_util_internal::call_functor_with_args_from_stack<FuncType>(
        ctx,
        stack,
        std::make_index_sequence<num_inputs>(),
        static_cast<ContextRemovedArgsType*>(nullptr));
  }
};

template <typename FuncType>
static executorch::runtime::Kernel make_boxed_kernel(
    const char* name,
    FuncType) {
  return executorch::runtime::Kernel(
      name, WrapUnboxedIntoFunctor<FuncType>::call);
}

} // namespace extension
} // namespace executorch

#define EXECUTORCH_LIBRARY(ns, op_name, func)                    \
  static auto res_##ns = ::executorch::runtime::register_kernel( \
      ::executorch::extension::make_boxed_kernel(                \
          #ns "::" op_name, EXECUTORCH_FN(func)))

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::make_boxed_kernel;
using ::executorch::extension::WrapUnboxedIntoFunctor;
} // namespace executor
} // namespace torch
