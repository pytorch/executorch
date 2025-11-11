/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/function_ref.h>

/// This header is DEPRECATED; use executorch/runtime/core/function_ref.h
/// directly instead.

namespace executorch::extension::pytree {
using executorch::runtime::FunctionRef;
} // namespace executorch::extension::pytree

namespace torch::executor::pytree {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::pytree::FunctionRef;
} // namespace torch::executor::pytree
