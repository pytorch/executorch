/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm

/**
 * Configuration class for initializing a LlmModule.
 *
 * Use [create] method and the fluent builder pattern.
 */
class LlmModuleConfig
private constructor(
    val modulePath: String,
    val tokenizerPath: String,
    val temperature: Float,
    val dataPath: String?,
    val modelType: Int,
    val numBos: Int,
    val numEos: Int,
    val loadMode: Int,
) {

  companion object {
    /** Load entire model file into a buffer (no mmap). */
    const val LOAD_MODE_FILE = 0

    /** Load model via mmap without mlock (default). Pages faulted in on demand. */
    const val LOAD_MODE_MMAP = 1

    /** Load model via mmap and pin all pages with mlock. */
    const val LOAD_MODE_MMAP_USE_MLOCK = 2

    /** Load model via mmap and attempt mlock, ignoring mlock failures. */
    const val LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS = 3

    /** Model type constant for text-only models. */
    const val MODEL_TYPE_TEXT = 1

    /** Model type constant for text-and-vision multimodal models. */
    const val MODEL_TYPE_TEXT_VISION = 2

    /** Model type constant for generic multimodal models. */
    const val MODEL_TYPE_MULTIMODAL = 2

    /**
     * Creates a new Builder instance for constructing LlmModuleConfig objects.
     *
     * @return a new Builder instance with default configuration values
     */
    @JvmStatic fun create(): Builder = Builder()
  }

  /**
   * Builder class for constructing LlmModuleConfig instances with optional parameters.
   *
   * The builder provides a fluent interface for configuring model parameters and validates required
   * fields before construction.
   */
  class Builder internal constructor() {
    internal var modulePath: String? = null
    internal var tokenizerPath: String? = null
    internal var temperature: Float = 0.8f
    internal var dataPath: String? = ""
    internal var modelType: Int = MODEL_TYPE_TEXT
    internal var numBos: Int = 0
    internal var numEos: Int = 0
    internal var loadMode: Int = LOAD_MODE_MMAP

    /** Sets the path to the module. */
    fun modulePath(modulePath: String): Builder = apply { this.modulePath = modulePath }

    /** Sets the path to the tokenizer. */
    fun tokenizerPath(tokenizerPath: String): Builder = apply {
      this.tokenizerPath = tokenizerPath
    }

    /** Sets the temperature for sampling generation. */
    fun temperature(temperature: Float): Builder = apply { this.temperature = temperature }

    /** Sets the path to optional additional data files. */
    fun dataPath(dataPath: String?): Builder = apply { this.dataPath = dataPath }

    /** Sets the model type (text-only or multimodal). */
    fun modelType(modelType: Int): Builder = apply { this.modelType = modelType }

    /** Sets the number of BOS tokens to prepend. */
    fun numBos(numBos: Int): Builder = apply { this.numBos = numBos }

    /** Sets the number of EOS tokens to append. */
    fun numEos(numEos: Int): Builder = apply { this.numEos = numEos }

    /**
     * Sets the load mode for the model file. Defaults to [LOAD_MODE_MMAP] (mmap without mlock),
     * which avoids pinning model pages in RAM.
     *
     * @throws IllegalArgumentException if loadMode is not one of the supported constants
     */
    fun loadMode(loadMode: Int): Builder {
      require(
          loadMode == LOAD_MODE_FILE ||
              loadMode == LOAD_MODE_MMAP ||
              loadMode == LOAD_MODE_MMAP_USE_MLOCK ||
              loadMode == LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS) {
            "Unknown load mode: $loadMode"
          }
      return apply { this.loadMode = loadMode }
    }

    /**
     * Constructs the LlmModuleConfig instance with validated parameters.
     *
     * @throws IllegalArgumentException if required fields are missing
     */
    fun build(): LlmModuleConfig {
      require(modulePath != null && tokenizerPath != null) {
        "Module path and tokenizer path are required"
      }
      return LlmModuleConfig(
          modulePath!!,
          tokenizerPath!!,
          temperature,
          dataPath,
          modelType,
          numBos,
          numEos,
          loadMode,
      )
    }
  }
}
