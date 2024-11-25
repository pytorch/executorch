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
};

} // namespace tokenizers
