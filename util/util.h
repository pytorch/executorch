/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * DEPRECATED: Do not use this file or add new functions to it.
 */

#pragma once

#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/executor/method.h>

namespace torch {
namespace executor {
namespace util {

/**
 * DEPRECATED: Use prepare_input_tensors() instead.
 *
 * Allocates input tensors for the provided Method, filling them with ones.
 *
 * @param[in] method The Method that owns the inputs to prepare.
 * @returns An array of pointers that must be passed to `FreeInputs()` after
 *     the Method is no longer needed.
 */
__ET_DEPRECATED
inline exec_aten::ArrayRef<void*> PrepareInputTensors(Method& method) {
  Result<BufferCleanup> inputs = prepare_input_tensors(method);
  ET_CHECK(inputs.ok());
  // A hack to work with the deprecated signature. Return an ArrayRef that
  // points to a single BufferCleanup.
  return {
      reinterpret_cast<void**>(new BufferCleanup(std::move(inputs.get()))), 1};
}

/**
 * DEPRECATED: Use prepare_input_tensors() instead, which does not need this.
 *
 * Frees memory that was allocated by `PrepareInputTensors()`.
 */
__ET_DEPRECATED
inline void FreeInputs(exec_aten::ArrayRef<void*> inputs) {
  ET_CHECK(inputs.size() == 1);
  // A hack to work with the deprecated signature. The ArrayRef points to a
  // single BufferCleanup for us to delete.
  delete reinterpret_cast<BufferCleanup*>(const_cast<void**>(inputs.data()));
}

} // namespace util
} // namespace executor
} // namespace torch
