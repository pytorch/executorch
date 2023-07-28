/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Executorch global runtime wrapper functions.
 */

#pragma once

#include <executorch/runtime/platform/compiler.h>

namespace torch {
namespace executor {

/**
 * Initialize the Executorch global runtime.
 */
void runtime_init();

} // namespace executor
} // namespace torch
