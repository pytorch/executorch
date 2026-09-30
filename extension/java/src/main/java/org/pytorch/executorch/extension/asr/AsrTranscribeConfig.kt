/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.asr

import org.pytorch.executorch.annotations.Experimental

/**
 * Configuration for ASR transcription.
 *
 * Warning: These APIs are experimental and subject to change without notice
 *
 * @property maxNewTokens Maximum number of new tokens to generate (must be positive)
 * @property temperature Temperature for sampling. 0.0 means greedy decoding
 * @property decoderStartTokenId The token ID to start decoding with (e.g., language token for
 *   Whisper)
 */
@Experimental
data class AsrTranscribeConfig(
    val maxNewTokens: Long = 128,
    val temperature: Float = 0.0f,
    val decoderStartTokenId: Long = 0,
) {
  init {
    require(maxNewTokens > 0) { "maxNewTokens must be positive" }
    require(temperature >= 0) { "temperature must be non-negative" }
  }

  /** Builder class for AsrTranscribeConfig for Java interoperability. */
  class Builder {
    private var maxNewTokens: Long = 128
    private var temperature: Float = 0.0f
    private var decoderStartTokenId: Long = 0

    fun setMaxNewTokens(maxNewTokens: Long) = apply {
      require(maxNewTokens > 0) { "maxNewTokens must be positive" }
      this.maxNewTokens = maxNewTokens
    }

    fun setTemperature(temperature: Float) = apply {
      require(temperature >= 0) { "temperature must be non-negative" }
      this.temperature = temperature
    }

    fun setDecoderStartTokenId(decoderStartTokenId: Long) = apply {
      this.decoderStartTokenId = decoderStartTokenId
    }

    fun build() =
        AsrTranscribeConfig(
            maxNewTokens = maxNewTokens,
            temperature = temperature,
            decoderStartTokenId = decoderStartTokenId,
        )
  }
}
