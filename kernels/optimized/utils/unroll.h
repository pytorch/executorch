/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/compiler.h>

// Utility to guaruntee complete unrolling of a loop where the bounds are known
// at compile time. Various pragmas achieve similar effects, but are not as
// portable across compilers.

// Example: ForcedUnroll<4>{}(f); is equivalent to f(0); f(1); f(2); f(3);

namespace executorch {
namespace utils {

template <int n>
struct ForcedUnroll {
  template <typename Func>
  ET_INLINE void operator()(const Func& f) const {
    ForcedUnroll<n - 1>{}(f);
    f(n - 1);
  }
};

template <>
struct ForcedUnroll<1> {
  template <typename Func>
  ET_INLINE void operator()(const Func& f) const {
    f(0);
  }
};

} // namespace utils
} // namespace executorch
