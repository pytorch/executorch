/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm;

/**
 * Configuration class for controlling text generation parameters in LLM operations.
 *
 * <p>This class provides settings for text generation behavior including output formatting,
 * generation limits, and sampling parameters. Instances should be created using the {@link
 * #create()} method and the fluent builder pattern.
 */
public class LlmGenerationConfig {
  private final boolean echo;
  private final int maxNewTokens;
  private final boolean warming;
  private final int seqLen;
  private final float temperature;
  private final int numBos;
  private final int numEos;

  private LlmGenerationConfig(Builder builder) {
    this.echo = builder.echo;
    this.maxNewTokens = builder.maxNewTokens;
    this.warming = builder.warming;
    this.seqLen = builder.seqLen;
    this.temperature = builder.temperature;
    this.numBos = builder.numBos;
    this.numEos = builder.numEos;
  }

  /**
   * Creates a new Builder instance for constructing generation configurations.
   *
   * @return a new Builder with default configuration values
   */
  public static Builder create() {
    return new Builder();
  }

  /**
   * @return true if input prompt should be included in the output
   */
  public boolean isEcho() {
    return echo;
  }

  /**
   * @return maximum number of tokens to generate (-1 for unlimited)
   */
  public int getMaxNewTokens() {
    return maxNewTokens;
  }

  /**
   * @return true if model warming is enabled
   */
  public boolean isWarming() {
    return warming;
  }

  /**
   * @return maximum sequence length for generation (-1 for default)
   */
  public int getSeqLen() {
    return seqLen;
  }

  /**
   * @return temperature value for sampling (higher = more random)
   */
  public float getTemperature() {
    return temperature;
  }

  /**
   * @return number of BOS tokens to prepend
   */
  public int getNumBos() {
    return numBos;
  }

  /**
   * @return number of EOS tokens to append
   */
  public int getNumEos() {
    return numEos;
  }

  /**
   * Builder class for constructing LlmGenerationConfig instances.
   *
   * <p>Provides a fluent interface for configuring generation parameters with sensible defaults.
   * All methods return the builder instance to enable method chaining.
   */
  public static class Builder {
    private boolean echo = true;
    private int maxNewTokens = -1;
    private boolean warming = false;
    private int seqLen = -1;
    private float temperature = 0.8f;
    private int numBos = 0;
    private int numEos = 0;

    Builder() {}

    /**
     * Sets whether to include the input prompt in the generated output.
     *
     * @param echo true to include input prompt, false to return only new tokens
     * @return this builder instance
     */
    public Builder echo(boolean echo) {
      this.echo = echo;
      return this;
    }

    /**
     * Sets the maximum number of new tokens to generate.
     *
     * @param maxNewTokens the token limit (-1 for unlimited generation)
     * @return this builder instance
     */
    public Builder maxNewTokens(int maxNewTokens) {
      this.maxNewTokens = maxNewTokens;
      return this;
    }

    /**
     * Enables or disables model warming.
     *
     * @param warming true to generate initial tokens for model warmup
     * @return this builder instance
     */
    public Builder warming(boolean warming) {
      this.warming = warming;
      return this;
    }

    /**
     * Sets the maximum sequence length for generation.
     *
     * @param seqLen maximum sequence length (-1 for default behavior)
     * @return this builder instance
     */
    public Builder seqLen(int seqLen) {
      this.seqLen = seqLen;
      return this;
    }

    /**
     * Sets the temperature for random sampling.
     *
     * @param temperature sampling temperature (typical range 0.0-1.0)
     * @return this builder instance
     */
    public Builder temperature(float temperature) {
      this.temperature = temperature;
      return this;
    }

    /**
     * Sets the number of BOS tokens to prepend.
     *
     * @param numBos number of BOS tokens
     * @return this builder instance
     */
    public Builder numBos(int numBos) {
      this.numBos = numBos;
      return this;
    }

    /**
     * Sets the number of EOS tokens to append.
     *
     * @param numEos number of EOS tokens
     * @return this builder instance
     */
    public Builder numEos(int numEos) {
      this.numEos = numEos;
      return this;
    }

    /**
     * Constructs the LlmGenerationConfig instance with the configured parameters.
     *
     * @return new LlmGenerationConfig instance with current builder settings
     */
    public LlmGenerationConfig build() {
      return new LlmGenerationConfig(this);
    }
  }
}
