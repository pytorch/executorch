/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string_view>

#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace runtime {
namespace etensor {

using string_view ET_DEPRECATED = std::string_view;

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using string_view ET_DEPRECATED = std::string_view;
} // namespace executor
} // namespace torch
