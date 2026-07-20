/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#if (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) || \
    (!defined(_MSVC_LANG) && __cplusplus >= 202002L)
namespace torch {
namespace executor {
namespace native {
namespace utils {
namespace internal {

template <std::size_t N>
struct OperatorName {
  char value[N];

  constexpr OperatorName(const char (&name)[N]) : value{} {
    for (std::size_t i = 0; i < N; ++i) {
      value[i] = name[i];
    }
  }

  constexpr operator const char*() const {
    return value;
  }
};

} // namespace internal
} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch

#define ET_OPERATOR_NAME_TYPE \
  ::torch::executor::native::utils::internal::OperatorName
#else
#define ET_OPERATOR_NAME_TYPE const char*
#endif
