/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * ExecuTorch global abort wrapper function.
 */

#pragma once

#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace runtime {

/**
 * Trigger the ExecuTorch global runtime to immediately exit without cleaning
 * up, and set an abnormal exit status (platform-defined).
 */
__ET_NORETURN void runtime_abort();

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::runtime_abort;
} // namespace executor
} // namespace torch
