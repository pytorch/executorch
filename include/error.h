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

  /// Artifact load failure.
  LoadFailure = 0x04,

  /// Encode failure.
  EncodeFailure = 0x05,

  /// Base64 decode failure.
  Base64DecodeFailure = 0x06,
};

} // namespace tokenizers

/**
 * If cond__ is false, return the specified Error
 * from the current function, which must be of return type
 * tokenizers::Error.
 * TODO: Add logging support
 * @param[in] cond__ The condition to be checked, asserted as true.
 * @param[in] error__ Error enum value to return without the `Error::` prefix,
 * like `InvalidArgument`.
 */
#define TK_CHECK_OR_RETURN_ERROR(cond__, error__)                              \
  {                                                                            \
    if (!(cond__)) {                                                           \
      return ::tokenizers::Error::error__;                                     \
    }                                                                          \
  }

/**
 * If error__ is not Error::Ok, return the specified Error
 * TODO: Add logging support
 * @param[in] error__ Error enum value to return without the `Error::` prefix,
 * like `InvalidArgument`.
 */
#define TK_CHECK_OK_OR_RETURN_ERROR(error__)                                   \
  {                                                                            \
    if (error__ != ::tokenizers::Error::Ok) {                                  \
      return error__;                                                          \
    }                                                                          \
  }
