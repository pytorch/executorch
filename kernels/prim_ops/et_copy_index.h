/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace torch {
namespace executor {
namespace function {

void et_copy_index(KernelRuntimeContext& context, EValue** stack);

} // namespace function
} // namespace executor
} // namespace torch
