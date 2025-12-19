/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Thin wrapper to reuse ExecuTorch's c10 llvmMathExtras implementation.
// This provides backward compatibility for SlimTensor code that uses
// executorch::backends::aoti::slim::c10::llvm functions.

#include <c10/util/llvmMathExtras.h>

namespace executorch {
namespace backends {
namespace aoti {
namespace slim {
namespace c10 {
namespace llvm {

using ::c10::llvm::Hi_32;
using ::c10::llvm::Lo_32;
using ::c10::llvm::Make_64;

} // namespace llvm
} // namespace c10
} // namespace slim
} // namespace aoti
} // namespace backends
} // namespace executorch
