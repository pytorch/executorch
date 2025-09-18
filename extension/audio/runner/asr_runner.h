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

#include <executorch/extension/llm/runner/audio.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/runtime/core/error.h>

namespace executorch {
namespace extension {
namespace audio {

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
   * @param audio processed audio input, which contains a vector of bytes that
   * encodes a float tensor in little-endian byte order
   * @param token_callback Callback function called for each generated token
   * @param stats_callback Callback function for generation statistics
   * @return Error::Ok if successful, an error otherwise
   */
  virtual runtime::Error transcribe(
      int32_t seq_len,
      ::executorch::extension::llm::Audio& audio,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const ::executorch::extension::llm::Stats&)>
          stats_callback = {}) = 0;
};

} // namespace audio
} // namespace extension
} // namespace executorch
