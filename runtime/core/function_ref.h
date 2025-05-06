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
// - renamed to FunctionRef
// - removed LLVM_GSL_POINTER and LLVM_LIFETIME_BOUND macro uses
// - use namespaced internal::remove_cvref_t

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

namespace executorch::runtime {

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
  Ret (*callback)(intptr_t callable, Params... params) = nullptr;
  intptr_t callable;

  template <typename Callable>
  static Ret callback_fn(intptr_t callable, Params... params) {
    return (*reinterpret_cast<Callable*>(callable))(
        std::forward<Params>(params)...);
  }

 public:
  FunctionRef() = default;
  FunctionRef(std::nullptr_t) {}

  template <typename Callable>
  FunctionRef(
      Callable&& callable,
      // This is not the copy-constructor.
      std::enable_if_t<!std::is_same<
          internal::remove_cvref_t<Callable>,
          FunctionRef>::value>* = nullptr,
      // Functor must be callable and return a suitable type.
      std::enable_if_t<
          std::is_void<Ret>::value ||
          std::is_convertible<
              decltype(std::declval<Callable>()(std::declval<Params>()...)),
              Ret>::value>* = nullptr)
      : callback(callback_fn<std::remove_reference_t<Callable>>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}

  Ret operator()(Params... params) const {
    return callback(callable, std::forward<Params>(params)...);
  }

  explicit operator bool() const {
    return callback;
  }

  bool operator==(const FunctionRef<Ret(Params...)>& Other) const {
    return callable == Other.callable;
  }
};
} // namespace executorch::runtime
