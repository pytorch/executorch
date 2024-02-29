/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
/// \file runtime/kernel/make_boxed_from_unboxed_functor.h
/// Defines a template that can be used to create a boxed version of an unboxed
/// functor.
/// Example usage:
/// ```
/// Tensor&
/// my_op(RuntimeContext& ctx, const Tensor& self, const Tensor& other, Tensor&
/// out) {
///   // ...
///   return out;
/// }
///
/// Kernel my_kernel = Kernel.make_boxed_kernel("my_ns::my_op",
/// EXECUTORCH_FN(my_op)); register_kernels({my_kernel});
/// ```
///
/// The trick here is to convert each EValue to inferred argument type. This
/// uses a lot of C++17 features.
//===----------------------------------------------------------------------===//

#pragma once
#if __cplusplus < 201703L
#error "This header requires C++17"
#endif

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/type_list.h>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <typeinfo>

namespace torch {
namespace executor {

class KernelRuntimeContext; // Forward declaration
using RuntimeContext = KernelRuntimeContext; // TODO(T147221312): Remove

// Check if a given type is a function
template <class T>
struct is_function_type : std::false_type {};
template <class Result, class... Args>
struct is_function_type<Result(Args...)> : std::true_type {};
template <class T>
using is_function_type_t = typename is_function_type<T>::type;

// A compile-time wrapper around a function pointer
template <class FuncType_, FuncType_* func_ptr_>
struct CompileTimeFunctionPointer final {
  static_assert(
      is_function_type<FuncType_>::value,
      "EXECUTORCH_FN can only wrap function types.");
  using FuncType = FuncType_;

  static constexpr FuncType* func_ptr() {
    return func_ptr_;
  }
};

// Check if a given type is a compile-time function pointer
template <class T>
struct is_compile_time_function_pointer : std::false_type {};
template <class FuncType, FuncType* func_ptr>
struct is_compile_time_function_pointer<
    CompileTimeFunctionPointer<FuncType, func_ptr>> : std::true_type {};

#define EXECUTORCH_FN_TYPE(func)                                      \
  CompileTimeFunctionPointer<                                         \
      std::remove_pointer_t<std::remove_reference_t<decltype(func)>>, \
      func>
#define EXECUTORCH_FN(func) EXECUTORCH_FN_TYPE(func)()

/**
 * strip_class: helper to remove the class type from pointers to `operator()`.
 */
template <typename T>
struct strip_class {};
template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...)> {
  using type = Result(Args...);
};
template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...) const> {
  using type = Result(Args...);
};
template <typename T>
using strip_class_t = typename strip_class<T>::type;

/**
 * Access information about result type or arguments from a function type.
 * Example:
 * using A = function_traits<int (float, double)>::return_type // A == int
 * using A = function_traits<int (float, double)>::parameter_types::tuple_type
 * // A == tuple<float, double>
 */
template <class Func>
struct function_traits {
  static_assert(
      !std::is_same<Func, Func>::value,
      "In function_traits<Func>, Func must be a plain function type.");
};
template <class Result, class... Args>
struct function_traits<Result(Args...)> {
  using func_type = Result(Args...);
  using return_type = Result;
  using parameter_types = typelist<Args...>;
  static constexpr auto number_of_parameters = sizeof...(Args);
};

/**
 * infer_function_traits: creates a `function_traits` type for a simple
 * function (pointer) or functor (lambda/struct). Currently does not support
 * class methods.
 */
template <typename Functor>
struct infer_function_traits {
  using type = function_traits<strip_class_t<decltype(&Functor::operator())>>;
};
template <typename Result, typename... Args>
struct infer_function_traits<Result (*)(Args...)> {
  using type = function_traits<Result(Args...)>;
};
template <typename Result, typename... Args>
struct infer_function_traits<Result(Args...)> {
  using type = function_traits<Result(Args...)>;
};
template <typename T>
using infer_function_traits_t = typename infer_function_traits<T>::type;

// evalue_to_arg
template <class T>
struct decay_if_not_tensor final {
  using type = std::decay_t<T>;
};
template <>
struct decay_if_not_tensor<exec_aten::Tensor&> final {
  using type = exec_aten::Tensor&;
};
template <>
struct decay_if_not_tensor<const exec_aten::Tensor&> final {
  using type = const exec_aten::Tensor&;
};

template <class T>
struct evalue_to_arg final {
  static T call(EValue& v) {
    return std::move(v).to<T>();
  }
};

template <>
struct evalue_to_arg<exec_aten::Tensor&> final {
  static exec_aten::Tensor& call(EValue& v) {
    return v.toTensor();
  }
};

template <>
struct evalue_to_arg<const exec_aten::Tensor&> final {
  static const exec_aten::Tensor& call(EValue& v) {
    return v.toTensor();
  }
};
// Call functor with args from stack

template <class Functor, size_t... evalue_arg_indices, typename... ArgTypes>
void call_functor_with_args_from_stack_(
    RuntimeContext& ctx,
    EValue** stack,
    std::index_sequence<evalue_arg_indices...>,
    typelist<ArgTypes...>*) {
  (*Functor::func_ptr())(
      ctx,
      evalue_to_arg<typename decay_if_not_tensor<ArgTypes>::type>::call(
          *stack[evalue_arg_indices])...);
}

/**
 * WrapUnboxedIntoFunctor: Given a function pointer, wrap it into a functor that
 * takes EValues as input and returns void. The wrapped functor will unbox all
 * inputs and forward them to unboxed kernel.
 */
template <class FuncType>
struct WrapUnboxedIntoFunctor {
  static_assert(
      is_compile_time_function_pointer<FuncType>::value,
      "Can't handle function other than EXECUTORCH_FN");
  using TrueType = typename FuncType::FuncType;
  using ReturnType = typename infer_function_traits_t<TrueType>::return_type;
  using ArgsType = typename infer_function_traits_t<TrueType>::parameter_types;
  // check if the first argument is RuntimeContext, if so, remove it
  static constexpr bool first_arg_is_context = std::is_same<
      RuntimeContext,
      std::remove_reference_t<head_with_default_t<void, ArgsType>>>::value;
  using ContextRemovedArgsType = std::conditional_t<
      first_arg_is_context,
      drop_if_nonempty_t<ArgsType, 1>,
      ArgsType>;

  static void call(RuntimeContext& ctx, EValue** stack) {
    constexpr size_t num_inputs = size<ContextRemovedArgsType>::value;
    return call_functor_with_args_from_stack_<FuncType>(
        ctx,
        stack,
        std::make_index_sequence<num_inputs>(),
        static_cast<ContextRemovedArgsType*>(nullptr));
  }
};

} // namespace executor
} // namespace torch
