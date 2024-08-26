/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===- llvm/ADT/STLFunctionalExtras.h - Extras for <functional> -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some extension to <functional>.
//
// No library is required when using these functions.
//
//===----------------------------------------------------------------------===//
//     Extra additions to <functional>
//===----------------------------------------------------------------------===//

/// An efficient, type-erasing, non-owning reference to a callable. This is
/// intended for use as the type of a function parameter that is not used
/// after the function in question returns.
///
/// This class does not own the callable, so it is not in general safe to store
/// a FunctionRef.

// torch::executor: modified from llvm::function_ref
// see https://www.foonathan.net/2017/01/function-ref-implementation/

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

namespace executorch {
namespace extension {
namespace pytree {

//===----------------------------------------------------------------------===//
//     Features from C++20
//===----------------------------------------------------------------------===//

namespace internal {

template <typename T>
struct remove_cvref {
  using type =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

} // namespace internal

template <typename Fn>
class FunctionRef;

template <typename Ret, typename... Params>
class FunctionRef<Ret(Params...)> {
  Ret (*callback_)(const void* memory, Params... params) = nullptr;
  union Storage {
    void* callable;
    Ret (*function)(Params...);
  } storage_;

 public:
  FunctionRef() = default;
  explicit FunctionRef(std::nullptr_t) {}

  /**
   * Case 1: A callable object passed by lvalue reference.
   * Taking rvalue reference is error prone because the object will be always
   * be destroyed immediately.
   */
  template <
      typename Callable,
      // This is not the copy-constructor.
      typename std::enable_if<
          !std::is_same<internal::remove_cvref_t<Callable>, FunctionRef>::value,
          int32_t>::type = 0,
      // Avoid lvalue reference to non-capturing lambda.
      typename std::enable_if<
          !std::is_convertible<Callable, Ret (*)(Params...)>::value,
          int32_t>::type = 0,
      // Functor must be callable and return a suitable type.
      // To make this container type safe, we need to ensure either:
      // 1. The return type is void.
      // 2. Or the resulting type from calling the callable is convertible to
      // the declared return type.
      typename std::enable_if<
          std::is_void<Ret>::value ||
              std::is_convertible<
                  decltype(std::declval<Callable>()(std::declval<Params>()...)),
                  Ret>::value,
          int32_t>::type = 0>
  explicit FunctionRef(Callable& callable)
      : callback_([](const void* memory, Params... params) {
          auto& storage = *static_cast<const Storage*>(memory);
          auto& callable = *static_cast<Callable*>(storage.callable);
          return static_cast<Ret>(callable(std::forward<Params>(params)...));
        }) {
    storage_.callable = &callable;
  }

  /**
   * Case 2: A plain function pointer.
   * Instead of storing an opaque pointer to underlying callable object,
   * store a function pointer directly.
   * Note that in the future a variant which coerces compatible function
   * pointers could be implemented by erasing the storage type.
   */
  /* implicit */ FunctionRef(Ret (*ptr)(Params...))
      : callback_([](const void* memory, Params... params) {
          auto& storage = *static_cast<const Storage*>(memory);
          return storage.function(std::forward<Params>(params)...);
        }) {
    storage_.function = ptr;
  }

  /**
   * Case 3: Implicit conversion from lambda to FunctionRef.
   * A common use pattern is like:
   * void foo(FunctionRef<...>) {...}
   * foo([](...){...})
   * Here constructors for non const lvalue reference or function pointer
   * would not work because they do not cover implicit conversion from rvalue
   * lambda.
   * We need to define a constructor for capturing temporary callables and
   * always try to convert the lambda to a function pointer behind the scene.
   */
  template <
      typename Function,
      // This is not the copy-constructor.
      typename std::enable_if<
          !std::is_same<Function, FunctionRef>::value,
          int32_t>::type = 0,
      // Function is convertible to pointer of (Params...) -> Ret.
      typename std::enable_if<
          std::is_convertible<Function, Ret (*)(Params...)>::value,
          int32_t>::type = 0>
  /* implicit */ FunctionRef(const Function& function)
      : FunctionRef(static_cast<Ret (*)(Params...)>(function)) {}

  Ret operator()(Params... params) const {
    return callback_(&storage_, std::forward<Params>(params)...);
  }

  explicit operator bool() const {
    return callback_;
  }
};

} // namespace pytree
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace pytree {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::pytree::FunctionRef;
} // namespace pytree
} // namespace executor
} // namespace torch
