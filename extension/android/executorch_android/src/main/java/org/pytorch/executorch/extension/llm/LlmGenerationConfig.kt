/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm

/**
 * Configuration class for controlling text generation parameters in LLM operations.
 *
 * This class provides settings for text generation behavior including output formatting, generation
 * limits, and sampling parameters. Instances should be created using the [create] method and the
 * fluent builder pattern.
 */
class LlmGenerationConfig
private constructor(
    @get:JvmName("isEcho") val echo: Boolean,
    val maxNewTokens: Int,
    @get:JvmName("isWarming") val warming: Boolean,
    val seqLen: Int,
    val temperature: Float,
    val numBos: Int,
    val numEos: Int,
) {

  companion object {
    /**
     * Creates a new Builder instance for constructing generation configurations.
     *
     * @return a new Builder with default configuration values
     */
    @JvmStatic fun create(): Builder = Builder()
  }

  /**
   * Builder class for constructing LlmGenerationConfig instances.
   *
   * Provides a fluent interface for configuring generation parameters with sensible defaults. All
   * methods return the builder instance to enable method chaining.
   */
  class Builder internal constructor() {
    private var echo: Boolean = true
    private var maxNewTokens: Int = -1
    private var warming: Boolean = false
    private var seqLen: Int = -1
    private var temperature: Float = 0.8f
    private var numBos: Int = 0
    private var numEos: Int = 0

    /** Sets whether to include the input prompt in the generated output. */
    fun echo(echo: Boolean): Builder = apply { this.echo = echo }

    /** Sets the maximum number of new tokens to generate. */
    fun maxNewTokens(maxNewTokens: Int): Builder = apply { this.maxNewTokens = maxNewTokens }

    /** Enables or disables model warming. */
    fun warming(warming: Boolean): Builder = apply { this.warming = warming }

    /** Sets the maximum sequence length for generation. */
    fun seqLen(seqLen: Int): Builder = apply { this.seqLen = seqLen }

    /** Sets the temperature for random sampling. */
    fun temperature(temperature: Float): Builder = apply { this.temperature = temperature }

    /** Sets the number of BOS tokens to prepend. */
    fun numBos(numBos: Int): Builder = apply { this.numBos = numBos }

    /** Sets the number of EOS tokens to append. */
    fun numEos(numEos: Int): Builder = apply { this.numEos = numEos }

    /** Constructs the LlmGenerationConfig instance with the configured parameters. */
    fun build(): LlmGenerationConfig =
        LlmGenerationConfig(echo, maxNewTokens, warming, seqLen, temperature, numBos, numEos)
  }
}
