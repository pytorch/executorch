/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Common includes used by all kernel implementations.
 */

#pragma once

#include <cstddef>

// This list should be very conservative since most kernel .cpp files will
// include these and depend on their transitive deps. Only add a header if 99%
// of kernels would have included it anyway.
#include <executorch/runtime/core/exec_aten/exec_aten.h> // IWYU pragma: export
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h> // IWYU pragma: export
#include <executorch/runtime/core/exec_aten/util/tensor_util.h> // IWYU pragma: export
#include <executorch/runtime/kernel/kernel_runtime_context.h> // IWYU pragma: export

#if defined(_MSC_VER) && defined(_MSVC_LANG) && _MSVC_LANG >= 202002L
#define ET_USE_STRUCTURAL_OPERATOR_NAME 1
namespace executorch {
namespace runtime {

// MSVC rejects pointers to function-local arrays as template arguments. Copy
// the operator name into a C++20 structural type instead.
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

} // namespace runtime
} // namespace executorch

#define ET_OPERATOR_NAME_TYPE ::executorch::runtime::OperatorName
#define ET_DEFINE_OPERATOR_NAME(variable, name)                          \
  static constexpr auto variable = ::executorch::runtime::OperatorName { \
    name                                                                 \
  }
#else
#define ET_USE_STRUCTURAL_OPERATOR_NAME 0
#define ET_OPERATOR_NAME_TYPE const char*
#define ET_DEFINE_OPERATOR_NAME(variable, name) \
  static constexpr const char variable[] = name
#endif
