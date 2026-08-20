/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace runtime {
namespace etensor {

template <typename T>
using optional ET_DEPRECATED = std::optional<T>;
using nullopt_t ET_DEPRECATED = std::nullopt_t;
ET_DEPRECATED inline constexpr std::nullopt_t nullopt{std::nullopt};

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
template <typename T>
using optional ET_DEPRECATED = std::optional<T>;
using nullopt_t ET_DEPRECATED = std::nullopt_t;
ET_DEPRECATED inline constexpr std::nullopt_t nullopt{std::nullopt};
} // namespace executor
} // namespace torch
