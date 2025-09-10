/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Interface for audio-to-text model runners. Currently only used for
// supporting QNN Whisper Runner

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/runtime/core/error.h>

namespace executorch {
namespace extension {
namespace llm {

class ET_EXPERIMENTAL ASRRunner {
 public:
  virtual ~ASRRunner() = default;

  /**
   * Check if the runner is loaded and ready for inference.
   *
   * @return true if the runner is loaded, false otherwise
   */
  virtual bool is_loaded() const = 0;

  /**
   * Load the model and prepare for inference.
   *
   * @return Error::Ok if successful, an error otherwise
   */
  virtual runtime::Error load() = 0;

  /**
   * Generate text from raw audio.
   *
   * @param seq_len Length of input sequence
   * @param inputs A vector containing one element: a vector of bytes that
   * encodes a float tensor in little-endian byte order
   * @param token_callback Callback function called for each generated token
   * @return Error::Ok if successful, an error otherwise
   */
  virtual runtime::Error transcribe(
      int32_t seq_len,
      std::vector<std::vector<char>>& inputs,
      std::function<void(const std::string&)> token_callback = {}) = 0;
};

} // namespace llm
} // namespace extension
} // namespace executorch
