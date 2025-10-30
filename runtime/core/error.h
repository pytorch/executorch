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

namespace executorch {
namespace runtime {

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

  /// Registration error: Exceeding the maximum number of kernels.
  RegistrationExceedingMaxKernels = 0x15,

  /// Registration error: The kernel is already registered.
  RegistrationAlreadyRegistered = 0x16,

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

  /// Error caused by the contents of external data.
  InvalidExternalData = 0x24,

  /// Does not have enough resources to perform the requested operation.
  OutOfResources = 0x25,

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

// Stringify the Error enum.
constexpr const char* to_string(const Error error) {
  switch (error) {
    case Error::Ok:
      return "Error::Ok";
    case Error::Internal:
      return "Error::Internal";
    case Error::InvalidState:
      return "Error::InvalidState";
    case Error::EndOfMethod:
      return "Error::EndOfMethod";
    case Error::NotSupported:
      return "Error::NotSupported";
    case Error::NotImplemented:
      return "Error::NotImplemented";
    case Error::InvalidArgument:
      return "Error::InvalidArgument";
    case Error::InvalidType:
      return "Error::InvalidType";
    case Error::OperatorMissing:
      return "Error::OperatorMissing";
    case Error::NotFound:
      return "Error::NotFound";
    case Error::MemoryAllocationFailed:
      return "Error::MemoryAllocationFailed";
    case Error::AccessFailed:
      return "Error::AccessFailed";
    case Error::InvalidProgram:
      return "Error::InvalidProgram";
    case Error::InvalidExternalData:
      return "Error::InvalidExternalData";
    case Error::OutOfResources:
      return "Error::OutOfResources";
    case Error::DelegateInvalidCompatibility:
      return "Error::DelegateInvalidCompatibility";
    case Error::DelegateMemoryAllocationFailed:
      return "Error::DelegateMemoryAllocationFailed";
    case Error::DelegateInvalidHandle:
      return "Error::DelegateInvalidHandle";
    case Error::RegistrationExceedingMaxKernels:
      return "Error::RegistrationExceedingMaxKernels";
    case Error::RegistrationAlreadyRegistered:
      return "Error::RegistrationAlreadyRegistered";
  }
}

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::Error;
using ::executorch::runtime::error_code_t;
} // namespace executor
} // namespace torch

/**
 * If cond__ is false, log the specified message and return the specified Error
 * from the current function, which must be of return type
 * executorch::runtime::Error.
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
      return ::executorch::runtime::Error::error__;               \
    }                                                             \
  }

/**
 * A convenience macro to be used in utility functions that check whether input
 * tensor(s) are valid, which are expected to return a boolean. Checks whether
 * `cond` is true; if not, log the failed check with `message` and return false.
 *
 * @param[in] cond the condition to check
 * @param[in] message an additional message to log with `cond`
 */
#define ET_CHECK_OR_RETURN_FALSE(cond__, message__, ...)                      \
  {                                                                           \
    if (!(cond__)) {                                                          \
      ET_LOG(Error, "Check failed (%s): " message__, #cond__, ##__VA_ARGS__); \
      return false;                                                           \
    }                                                                         \
  }

/**
 * If error__ is not Error::Ok, optionally log a message and return the error
 * from the current function, which must be of return type
 * executorch::runtime::Error.
 *
 * @param[in] error__ Error enum value asserted to be Error::Ok.
 * @param[in] ... Optional format string for the log error message and its
 * arguments.
 */
#define ET_CHECK_OK_OR_RETURN_ERROR(...) \
  ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR(__VA_ARGS__)

/**
 * Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
 * This macro selects the correct version of
 * ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR based on the number of arguments passed.
 * It uses a helper that reliably picks the 1-arg or 2+-arg form on
 * MSVC/Clang/GCC.
 */
#define ET_INTERNAL_EXPAND(x) x
#define ET_INTERNAL_GET_MACRO(                          \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, NAME, ...) \
  NAME

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
// Picks _2 for 2..10 args, _1 for exactly 1 arg.
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR(...)      \
  ET_INTERNAL_EXPAND(ET_INTERNAL_GET_MACRO(            \
      __VA_ARGS__,                                     \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2, /* 10 */ \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2, /* 9  */ \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2, /* 8  */ \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2, /* 7  */ \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2, /* 6  */ \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2, /* 5  */ \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2, /* 4  */ \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2, /* 3  */ \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2, /* 2  */ \
      ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_1 /* 1  */  \
      )(__VA_ARGS__))

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_1(error__)   \
  do {                                                    \
    const auto et_error__ = (error__);                    \
    if (et_error__ != ::executorch::runtime::Error::Ok) { \
      return et_error__;                                  \
    }                                                     \
  } while (0)

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define ET_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2(error__, message__, ...) \
  do {                                                                  \
    const auto et_error__ = (error__);                                  \
    if (et_error__ != ::executorch::runtime::Error::Ok) {               \
      ET_LOG(Error, message__, ##__VA_ARGS__);                          \
      return et_error__;                                                \
    }                                                                   \
  } while (0)
