/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Tokenizers Error declarations.
 */

#pragma once

#include <pytorch/tokenizers/log.h>
#include <stdint.h>

namespace tokenizers {

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

  /// Tokenizer uninitialized.
  Uninitialized = 0x02,

  /// Token out of range.
  OutOfRange = 0x03,

  /// Tokenizer artifact load failure.
  LoadFailure = 0x04,

  /// Encode failure.
  EncodeFailure = 0x05,

  /// Base64 decode failure.
  Base64DecodeFailure = 0x06,

  /// Failed to parse tokenizer artifact.
  ParseFailure = 0x07,

  /// Decode failure.
  DecodeFailure = 0x08,
};

} // namespace tokenizers

/**
 * If cond__ is false, return the specified Error
 * from the current function, which must be of return type
 * tokenizers::Error.
 * TODO: Add logging support
 * @param[in] cond__ The condition to be checked, asserted as true.
 * @param[in] error__ Error enum value to return without the `Error::` prefix,
 * like `Base64DecodeFailure`.
 * @param[in] message__ Format string for the log error message.
 * @param[in] ... Optional additional arguments for the format string.
 */
#define TK_CHECK_OR_RETURN_ERROR(cond__, error__, message__, ...) \
  {                                                               \
    if (!(cond__)) {                                              \
      TK_LOG(Error, message__, ##__VA_ARGS__);                    \
      return ::tokenizers::Error::error__;                        \
    }                                                             \
  }

/**
 * If error__ is not Error::Ok, return the specified Error
 * TODO: Add logging support
 * @param[in] error__ Error enum value to return without the `Error::` prefix,
 * like `Base64DecodeFailure`.
 * @param[in] ... Optional format string for the log error message and its
 * arguments.
 */
#define TK_CHECK_OK_OR_RETURN_ERROR(error__, ...) \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR(error__, ##__VA_ARGS__)

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR(...) \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_SELECT(    \
      __VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1) \
  (__VA_ARGS__)

/**
 * Internal only: Use TK_CHECK_OK_OR_RETURN_ERROR() instead.
 * This macro selects the correct version of
 * TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR based on the number of arguments passed.
 * It uses a trick with the preprocessor to count the number of arguments and
 * then selects the appropriate macro.
 *
 * The macro expansion uses __VA_ARGS__ to accept any number of arguments and
 * then appends them to TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_, followed by the
 * count of arguments. The count is determined by the macro
 * TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_SELECT which takes the arguments and
 * passes them along with a sequence of numbers (2, 1). The preprocessor then
 * matches this sequence to the correct number of arguments provided.
 *
 * If two arguments are passed, TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2 is
 * selected, suitable for cases where an error code and a custom message are
 * provided. If only one argument is passed,
 * TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_1 is selected, which is used for cases
 * with just an error code.
 *
 * Usage:
 * TK_CHECK_OK_OR_RETURN_ERROR(error_code); // Calls v1
 * TK_CHECK_OK_OR_RETURN_ERROR(error_code, "Error message", ...); // Calls v2
 */
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_SELECT( \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_##N

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_1(error__) \
  do {                                                  \
    const auto et_error__ = (error__);                  \
    if (et_error__ != ::tokenizers::Error::Ok) {        \
      return et_error__;                                \
    }                                                   \
  } while (0)

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2(error__, message__, ...) \
  do {                                                                  \
    const auto et_error__ = (error__);                                  \
    if (et_error__ != ::tokenizers::Error::Ok) {                        \
      TK_LOG(Error, message__, ##__VA_ARGS__);                          \
      return et_error__;                                                \
    }                                                                   \
  } while (0)

// Internal only: Use ET_CHECK_OK_OR_RETURN_ERROR() instead.
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_3 \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_4 \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_5 \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_6 \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_7 \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_8 \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_9 \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
#define TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_10 \
  TK_INTERNAL_CHECK_OK_OR_RETURN_ERROR_2
