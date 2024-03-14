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
/// my_op(RuntimeContext& ctx, const Tensor& self, const Tensor& other, Tensor&
/// out) {
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

namespace torch {
namespace executor {

class KernelRuntimeContext; // Forward declaration
using RuntimeContext = KernelRuntimeContext; // TODO(T147221312): Remove

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

template <typename FuncType>
static Kernel make_boxed_kernel(const char* name, FuncType) {
  return Kernel(name, WrapUnboxedIntoFunctor<FuncType>::call);
}

#define EXECUTORCH_LIBRARY(ns, op_name, func) \
  static auto res_##ns = register_kernels(    \
      make_boxed_kernel(#ns "::" op_name, EXECUTORCH_FN(func)))
} // namespace executor
} // namespace torch
