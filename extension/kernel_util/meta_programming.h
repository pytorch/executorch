/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/kernel_util/type_list.h>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <typeinfo>

namespace executorch {
namespace extension {
// This extension has a lot of generic internal names like "size"; use a unique
// internal namespace to avoid conflicts with other extensions.
namespace kernel_util_internal {

// Check if a given type is a function
template <class T>
struct is_function_type : std::false_type {};
template <class Result, class... Args>
struct is_function_type<Result(Args...)> : std::true_type {};
template <class T>
using is_function_type_t = typename is_function_type<T>::type;

// A compile-time wrapper around a function pointer
template <class FuncType_, FuncType_* func_ptr_, const char* func_name>
struct CompileTimeFunctionPointer final {
  static_assert(
      is_function_type<FuncType_>::value,
      "EXECUTORCH_FN can only wrap function types.");
  using FuncType = FuncType_;
  static constexpr const char* func_name_ = func_name;

  static constexpr FuncType* func_ptr() {
    return func_ptr_;
  }
};

// Check if a given type is a compile-time function pointer
template <class T>
struct is_compile_time_function_pointer : std::false_type {};
template <class FuncType, FuncType* func_ptr, const char* func_name>
struct is_compile_time_function_pointer<
    CompileTimeFunctionPointer<FuncType, func_ptr, func_name>>
    : std::true_type {};

#define EXECUTORCH_FN_TYPE(func, name)                                       \
  ::executorch::extension::kernel_util_internal::CompileTimeFunctionPointer< \
      std::remove_pointer_t<std::remove_reference_t<decltype(func)>>,        \
      func,                                                                  \
      name>
#define EXECUTORCH_FN(func, name) EXECUTORCH_FN_TYPE(func, name)()

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

} // namespace kernel_util_internal
} // namespace extension
} // namespace executorch
