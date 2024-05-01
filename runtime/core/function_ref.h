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

namespace torch {
namespace executor {
namespace internal {
template <typename Fn>
struct FunctionRefImpl;

template <typename Ret, typename... Params>
struct FunctionRefImpl<Ret(Params...)> {
  using type = Ret (*)(Params...);
};
} // namespace internal
template <typename Fn>
using FunctionRef = typename internal::FunctionRefImpl<Fn>::type;
} // namespace executor
} // namespace torch
