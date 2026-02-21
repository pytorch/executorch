/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>

// Forward declare flatbuffer types.
namespace executorch_flatbuffer {
struct ExecutionPlan;
struct Program;
struct Tensor;
struct TensorList;
} // namespace executorch_flatbuffer

namespace executorch {
namespace runtime {

/**
 * Validates that computing numel (number of elements) from the tensor's sizes
 * will not overflow. This check should be performed before creating TensorImpl
 * objects to prevent undefined behavior from integer overflow.
 *
 * @param[in] tensor The flatbuffer Tensor to validate.
 * @return Error::Ok if the numel calculation is safe, Error::InvalidProgram
 *     if computing numel would overflow.
 */
ET_NODISCARD Error validate_tensor(const executorch_flatbuffer::Tensor* tensor);

/**
 * Performs validation of all tensors and lists in the program, checking that
 * their metadata is semantically valid and will not cause issues during
 * execution.
 *
 * Currently validates:
 * - Tensor numel overflow (all tensors)
 * - TensorList element types (all TensorLists)
 *
 * @param[in] program The flatbuffer Program to validate.
 * @return Error::Ok if validation passes, Error::InvalidProgram if any
 *     validation check fails.
 */
ET_NODISCARD Error
validate_program(const executorch_flatbuffer::Program* program);

} // namespace runtime
} // namespace executorch
