/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * ExecuTorch Error declarations.
 */

#pragma once

#include <stdint.h>

#include <executorch/runtime/platform/log.h>

namespace torch {
namespace executor {

// Alias error code integral type to minimal platform width (32-bits for now).
typedef uint32_t error_code_t;

/**
 * ExecuTorch Error type.
 */
enum class Error : error_code_t {
  /*
   * System errors.
   */

  /// Status indicating a successful operation.
  Ok = 0x00,

  /// An internal error occurred.
  Internal = 0x01,

  /// Status indicating the executor is in an invalid state for a target
  /// operation
  InvalidState = 0x2,

  /// Status indicating there are no more steps of execution to run
  EndOfMethod = 0x03,

  /*
   * Logical errors.
   */

  /// Operation is not supported in the current context.
  NotSupported = 0x10,

  /// Operation is not yet implemented.
  NotImplemented = 0x11,

  /// User provided an invalid argument.
  InvalidArgument = 0x12,

  /// Object is an invalid type for the operation.
  InvalidType = 0x13,

  /// Operator(s) missing in the operator registry.
  OperatorMissing = 0x14,

  /*
   * Resource errors.
   */

  /// Requested resource could not be found.
  NotFound = 0x20,

  /// Could not allocate the requested memory.
  MemoryAllocationFailed = 0x21,

  /// Could not access a resource.
  AccessFailed = 0x22,

  /// Error caused by the contents of a program.
  InvalidProgram = 0x23,

  /*
   * Delegate errors.
   */

  /// Init stage: Backend receives an incompatible delegate version.
  DelegateInvalidCompatibility = 0x30,
  /// Init stage: Backend fails to allocate memory.
  DelegateMemoryAllocationFailed = 0x31,
  /// Execute stage: The handle is invalid.
  DelegateInvalidHandle = 0x32,

};

} // namespace executor
} // namespace torch

/**
 * If cond__ is false, log message__ and return the Error
 * from the current function, which must be declared to return
 * torch::executor::Error
 *
 * @param[in] cond__ Condition asserted as true
 * @param[in] error__ Error enum value like `InvalidArgument`.
 * @param[in] message__ Log error message format string.
 */
#define ET_CHECK_OR_RETURN_ERROR(cond__, error__, message__, ...) \
  ({                                                              \
    if (!(cond__)) {                                              \
      ET_LOG(Error, message__, ##__VA_ARGS__);                    \
      return torch::executor::Error::error__;                     \
    }                                                             \
  })

/**
 * If error__ is not Error::Ok, log message__ and return the Error
 * from the current function, which must be declared to return
 * torch::executor::Error
 *
 * @param[in] error__ Error enum value asserted to be Error::Ok.
 * @param[in] message__ Log error message format string.
 */
#define ET_CHECK_OK_OR_RETURN_ERROR(error__, message__, ...) \
  ({                                                         \
    if ((error__) != Error::Ok) {                            \
      ET_LOG(Error, message__, ##__VA_ARGS__);               \
      return error__;                                        \
    }                                                        \
  })
