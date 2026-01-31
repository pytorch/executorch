/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm;

/**
 * Configuration class for initializing a LlmModule.
 *
 * <p>{@link #create()} method and the fluent builder pattern.
 */
public class LlmModuleConfig {
  private final String modulePath;
  private final String tokenizerPath;
  private final float temperature;
  private final String dataPath;
  private final int modelType;
  private final int numBos;
  private final int numEos;

  private LlmModuleConfig(Builder builder) {
    this.modulePath = builder.modulePath;
    this.tokenizerPath = builder.tokenizerPath;
    this.temperature = builder.temperature;
    this.dataPath = builder.dataPath;
    this.modelType = builder.modelType;
    this.numBos = builder.numBos;
    this.numEos = builder.numEos;
  }

  /** Model type constant for text-only models. */
  public static final int MODEL_TYPE_TEXT = 1;

  /** Model type constant for text-and-vision multimodal models. */
  public static final int MODEL_TYPE_TEXT_VISION = 2;

  /** Model type constant for generic multimodal models. */
  public static final int MODEL_TYPE_MULTIMODAL = 2;

  /**
   * Creates a new Builder instance for constructing LlmModuleConfig objects.
   *
   * @return a new Builder instance with default configuration values
   */
  public static Builder create() {
    return new Builder();
  }

  // Getters with documentation
  /**
   * @return Path to the compiled model module (.pte file)
   */
  public String getModulePath() {
    return modulePath;
  }

  /**
   * @return Path to the tokenizer file or directory
   */
  public String getTokenizerPath() {
    return tokenizerPath;
  }

  /**
   * @return Temperature value for sampling (higher = more random)
   */
  public float getTemperature() {
    return temperature;
  }

  /**
   * @return Optional path to additional data files
   */
  public String getDataPath() {
    return dataPath;
  }

  /**
   * @return Type of model (text-only or text-vision)
   */
  public int getModelType() {
    return modelType;
  }

  /**
   * @return Number of BOS tokens to prepend
   */
  public int getNumBos() {
    return numBos;
  }

  /**
   * @return Number of EOS tokens to append
   */
  public int getNumEos() {
    return numEos;
  }

  /**
   * Builder class for constructing LlmModuleConfig instances with optional parameters.
   *
   * <p>The builder provides a fluent interface for configuring model parameters and validates
   * required fields before construction.
   */
  public static class Builder {
    private String modulePath;
    private String tokenizerPath;
    private float temperature = 0.8f;
    private String dataPath = "";
    private int modelType = MODEL_TYPE_TEXT;
    private int numBos = 0;
    private int numEos = 0;

    Builder() {}

    /**
     * Sets the path to the module.
     *
     * @param modulePath Path to module
     * @return This builder instance for method chaining
     */
    public Builder modulePath(String modulePath) {
      this.modulePath = modulePath;
      return this;
    }

    /**
     * Sets the path to the tokenizer.
     *
     * @param tokenizerPath Path to tokenizer
     * @return This builder instance for method chaining
     */
    public Builder tokenizerPath(String tokenizerPath) {
      this.tokenizerPath = tokenizerPath;
      return this;
    }

    /**
     * Sets the temperature for sampling generation.
     *
     * @param temperature Temperature value (typical range 0.0-1.0)
     * @return This builder instance for method chaining
     */
    public Builder temperature(float temperature) {
      this.temperature = temperature;
      return this;
    }

    /**
     * Sets the path to optional additional data files.
     *
     * @param dataPath Path to supplementary data resources
     * @return This builder instance for method chaining
     */
    public Builder dataPath(String dataPath) {
      this.dataPath = dataPath;
      return this;
    }

    /**
     * Sets the model type (text-only or multimodal).
     *
     * @param modelType One of MODEL_TYPE_TEXT, MODEL_TYPE_TEXT_VISION, MODEL_TYPE_MULTIMODAL
     * @return This builder instance for method chaining
     */
    public Builder modelType(int modelType) {
      this.modelType = modelType;
      return this;
    }

    /**
     * Sets the number of BOS tokens to prepend.
     *
     * @param numBos number of BOS tokens
     * @return This builder instance for method chaining
     */
    public Builder numBos(int numBos) {
      this.numBos = numBos;
      return this;
    }

    /**
     * Sets the number of EOS tokens to append.
     *
     * @param numEos number of EOS tokens
     * @return This builder instance for method chaining
     */
    public Builder numEos(int numEos) {
      this.numEos = numEos;
      return this;
    }

    /**
     * Constructs the LlmModuleConfig instance with validated parameters.
     *
     * @return New LlmModuleConfig instance with configured values
     * @throws IllegalArgumentException if required fields are missing
     */
    public LlmModuleConfig build() {
      if (modulePath == null || tokenizerPath == null) {
        throw new IllegalArgumentException("Module path and tokenizer path are required");
      }
      return new LlmModuleConfig(this);
    }
  }
}
