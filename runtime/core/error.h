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

static inline const char* error_message(enum Error err) {
  switch ((int)err) {
    case 0x00:
      return "OK: Successful complextion.";

    case 0x01:
      return "Internal: An internal error occurred.";

    case 0x02:
      return "InvalidState: Status indicating the executor is in an invalid state.";

    case 0x03:
      return "EndOfMethod: Status indicating there are no more steps of execution to run";

    case 0x10:
      return "NotSupported: Operation is not supported in the current context.";

    case 0x11:
      return "NotImplemented: Operation is not yet implemented.";

    case 0x12:
      return "InvalidArgument: User provided an invalid argument.";

    case 0x13:
      return "InvalidType: Object is an invalid type for the operation.";

    case 0x14:
      return "OperatorMissing: Operator(s) missing in the operator registry.";

    case 0x20:
      return "NotFound: Requested resource could not be found.";

    case 0x21:
      return "MemoryAllocationFailed: ould not allocate the requested memory.";

    case 0x22:
      return "AccessFailed: Could not access a resource.";

    case 0x23:
      return "InvalidProgram: Error caused by the contents of a program.";

    case 0x30:
      return "DelegateInvalidCompatibility: Backend receives an incompatbledelegate version (Init Stage).";

    case 0x31:
      return "DelegateMemoryAllocationFailed: Backend fails to allocate memory (Init Stage).";

    case 0x32:
      return "DelegateInvalidHandle: The handle is invalid (Execute stage).";

    default:
      return "This should not happen";
  }

  return nullptr;
};

} // namespace executor
} // namespace torch

/**
 * If cond__ is false, log the specified message and return the specified Error
 * from the current function, which must be of return type
 * torch::executor::Error.
 *
 * @param[in] cond__ The condition to be checked, asserted as true.
 * @param[in] error__ Error enum value to return without the `Error::` prefix,
 * like `InvalidArgument`.
 * @param[in] message__ Format string for the log error message.
 * @param[in] ... Optional additional arguments for the format string.
 */
#define ET_CHECK_OR_RETURN_ERROR(cond__, error__, message__, ...) \
  {                                                               \
    if (!(cond__)) {                                              \
      ET_LOG(Error, message__, ##__VA_ARGS__);                    \
      return ::torch::executor::Error::error__;                   \
    }                                                             \
  }

/**
 * If error__ is not Error::Ok, optionally log a message and return the error
 * from the current function, which must be of return type
 * torch::executor::Error.
 *
 * @param[in] error__ Error enum value asserted to be Error::Ok.
 * @param[in] ... Optional format string for the log error message and its
 * arguments.
 */
#define ET_CHECK_OK_OR_RETURN_ERROR(error__, ...) \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR(error__, ##__VA_ARGS__)

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR(...) \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_SELECT(    \
      __VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1) \
  (__VA_ARGS__)

/**
 * Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
 * This macro selects the correct version of
 * ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR based on the number of arguments passed.
 * It uses a trick with the preprocessor to count the number of arguments and
 * then selects the appropriate macro.
 *
 * The macro expansion uses __VA_ARGS__ to accept any number of arguments and
 * then appends them to ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_, followed by the
 * count of arguments. The count is determined by the macro
 * ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_SELECT which takes the arguments and
 * passes them along with a sequence of numbers (2, 1). The preprocessor then
 * matches this sequence to the correct number of arguments provided.
 *
 * If two arguments are passed, ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2 is
 * selected, suitable for cases where an error code and a custom message are
 * provided. If only one argument is passed,
 * ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_1 is selected, which is used for cases
 * with just an error code.
 *
 * Usage:
 * ET_CHECK_OK_OR_RETURN_ERROR(error_code); // Calls v1
 * ET_CHECK_OK_OR_RETURN_ERROR(error_code, "Error message", ...); // Calls v2
 */
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_SELECT( \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_##N

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_1(error__) \
  do {                                                  \
    const auto et_error__ = (error__);                  \
    if (et_error__ != ::torch::executor::Error::Ok) {   \
      return et_error__;                                \
    }                                                   \
  } while (0)

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2(error__, message__, ...) \
  do {                                                                  \
    const auto et_error__ = (error__);                                  \
    if (et_error__ != ::torch::executor::Error::Ok) {                   \
      ET_LOG(Error, message__, ##__VA_ARGS__);                          \
      return et_error__;                                                \
    }                                                                   \
  } while (0)

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_3 \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_4 \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_5 \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_6 \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_7 \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_8 \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_9 \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_10 \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
