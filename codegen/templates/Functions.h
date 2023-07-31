/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#pragma once

#include <tuple>

#include <executorch/runtime/core/exec_aten/exec_aten.h> // at::Tensor etc.
#include <executorch/codegen/macros.h> // TORCH_API
#include <executorch/runtime/kernel/kernel_runtime_context.h>

// ${generated_comment}

${static_dispatch_extra_headers}

namespace torch {
namespace executor {

${Functions_declarations}

} // namespace executor
} // namespace torch
