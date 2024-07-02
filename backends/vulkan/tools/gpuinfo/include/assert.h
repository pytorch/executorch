/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/PENGUINLIONG/graphi-t/blob/d291c3d1ce3795fe4b305e5efd76b4f586d23e3b/assert.hpp;
// MIT-licensed by Rendong Liang.

// Assertion.
// @PENGUINLIONG
#pragma once
#include "util.h"
#undef assert

namespace gpuinfo {

class AssertionFailedException : public std::exception {
  std::string msg;

 public:
  AssertionFailedException(const std::string& msg);

  const char* what() const noexcept override;
};

template <typename... TArgs>
inline void assert(bool pred, const TArgs&... args) {
  if (!pred) {
    throw AssertionFailedException(util::format(args...));
  }
}
template <typename... TArgs>
inline void panic(const TArgs&... args) {
  assert<TArgs...>(false, args...);
}
template <typename... TArgs>
inline void unreachable(const TArgs&... args) {
  assert<const char*, TArgs...>(false, "reached unreachable code: ", args...);
}

} // namespace gpuinfo
