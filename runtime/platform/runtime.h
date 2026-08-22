/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * ExecuTorch global runtime wrapper functions.
 */

#pragma once

#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace runtime {

/**
 * Initialize the ExecuTorch global runtime.
 */
void runtime_init();

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::runtime_init;
} // namespace executor
} // namespace torch
