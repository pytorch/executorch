/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm

import com.facebook.jni.HybridData
import com.facebook.jni.annotations.DoNotStrip
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.pytorch.executorch.ExecuTorchRuntime
import org.pytorch.executorch.annotations.Experimental

/**
 * LlmModule is a wrapper around the Executorch LLM. It provides a simple interface to generate text
 * from the model.
 *
 * Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
class LlmModule
private constructor(
    modelType: Int,
    modulePath: String?,
    tokenizerPath: String?,
    temperature: Float,
    dataFiles: List<String>,
    numBos: Int,
    numEos: Int,
    loadMode: Int,
) {

  private val mHybridData: HybridData

  init {
    ExecuTorchRuntime.getRuntime()
    requireNotNull(modulePath) { "model path must not be null" }
    requireNotNull(tokenizerPath) { "tokenizer path must not be null" }
    ExecuTorchRuntime.validateFilePath(modulePath, "model path")
    ExecuTorchRuntime.validateFilePath(tokenizerPath, "tokenizer path")
    mHybridData =
        initHybrid(
            modelType,
            modulePath,
            tokenizerPath,
            temperature,
            dataFiles,
            numBos,
            numEos,
            loadMode,
        )
  }

  /**
   * Constructs a LLM Module for a model with given type, model path, tokenizer, temperature, and
   * dataFiles.
   */
  constructor(
      modelType: Int,
      modulePath: String?,
      tokenizerPath: String?,
      temperature: Float,
      dataFiles: List<String>,
      numBos: Int,
      numEos: Int,
  ) : this(
      modelType,
      modulePath,
      tokenizerPath,
      temperature,
      dataFiles,
      numBos,
      numEos,
      DEFAULT_LOAD_MODE,
  )

  /**
   * Constructs a LLM Module for a model with given type, model path, tokenizer, temperature, and
   * dataFiles.
   */
  constructor(
      modelType: Int,
      modulePath: String?,
      tokenizerPath: String?,
      temperature: Float,
      dataFiles: List<String>,
  ) : this(
      modelType,
      modulePath,
      tokenizerPath,
      temperature,
      dataFiles,
      DEFAULT_BOS,
      DEFAULT_EOS,
      DEFAULT_LOAD_MODE,
  )

  /**
   * Constructs a LLM Module for a model with given type, model path, tokenizer, temperature, and
   * data path.
   */
  constructor(
      modelType: Int,
      modulePath: String?,
      tokenizerPath: String?,
      temperature: Float,
      dataPath: String?,
      numBos: Int,
      numEos: Int,
  ) : this(
      modelType,
      modulePath,
      tokenizerPath,
      temperature,
      listOfNotNull(dataPath),
      numBos,
      numEos,
  )

  /**
   * Constructs a LLM Module for a model with given type, model path, tokenizer, temperature, and
   * data path.
   */
  constructor(
      modelType: Int,
      modulePath: String?,
      tokenizerPath: String?,
      temperature: Float,
      dataPath: String?,
  ) : this(
      modelType,
      modulePath,
      tokenizerPath,
      temperature,
      dataPath,
      DEFAULT_BOS,
      DEFAULT_EOS,
  )

  /** Constructs a LLM Module for a model with given model path, tokenizer, temperature. */
  constructor(
      modulePath: String?,
      tokenizerPath: String?,
      temperature: Float,
  ) : this(
      MODEL_TYPE_TEXT,
      modulePath,
      tokenizerPath,
      temperature,
      emptyList(),
      DEFAULT_BOS,
      DEFAULT_EOS,
  )

  /**
   * Constructs a LLM Module for a model with given model path, tokenizer, temperature and data
   * path.
   */
  constructor(
      modulePath: String?,
      tokenizerPath: String?,
      temperature: Float,
      dataPath: String,
  ) : this(
      MODEL_TYPE_TEXT,
      modulePath,
      tokenizerPath,
      temperature,
      listOf(dataPath),
      DEFAULT_BOS,
      DEFAULT_EOS,
  )

  /** Constructs a LLM Module for a model with given path, tokenizer, and temperature. */
  constructor(
      modelType: Int,
      modulePath: String?,
      tokenizerPath: String?,
      temperature: Float,
  ) : this(
      modelType,
      modulePath,
      tokenizerPath,
      temperature,
      emptyList(),
      DEFAULT_BOS,
      DEFAULT_EOS,
  )

  /** Constructs a LLM Module for a model with the given LlmModuleConfig */
  constructor(config: LlmModuleConfig) : this(
      config.modelType,
      config.modulePath,
      config.tokenizerPath,
      config.temperature,
      listOfNotNull(config.dataPath),
      config.numBos,
      config.numEos,
      config.loadMode,
  )

  fun resetNative() {
    mHybridData.resetNative()
  }

  // --- generate overloads ---

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llmCallback callback object to receive results.
   */
  fun generate(prompt: String, llmCallback: LlmCallback): Int =
      generate(
          prompt,
          DEFAULT_SEQ_LEN,
          llmCallback,
          DEFAULT_ECHO,
          DEFAULT_TEMPERATURE,
          DEFAULT_BOS,
          DEFAULT_EOS,
      )

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results.
   */
  fun generate(prompt: String, seqLen: Int, llmCallback: LlmCallback): Int =
      generate(
          null,
          0,
          0,
          0,
          prompt,
          seqLen,
          llmCallback,
          DEFAULT_ECHO,
          DEFAULT_TEMPERATURE,
          DEFAULT_BOS,
          DEFAULT_EOS,
      )

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  fun generate(prompt: String, llmCallback: LlmCallback, echo: Boolean): Int =
      generate(
          null,
          0,
          0,
          0,
          prompt,
          DEFAULT_SEQ_LEN,
          llmCallback,
          echo,
          DEFAULT_TEMPERATURE,
          DEFAULT_BOS,
          DEFAULT_EOS,
      )

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  fun generate(prompt: String, seqLen: Int, llmCallback: LlmCallback, echo: Boolean): Int =
      generate(prompt, seqLen, llmCallback, echo, DEFAULT_TEMPERATURE, DEFAULT_BOS, DEFAULT_EOS)

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   * @param temperature temperature for sampling (use negative value to use module default)
   * @param numBos number of BOS tokens to prepend
   * @param numEos number of EOS tokens to append
   */
  external fun generate(
      prompt: String,
      seqLen: Int,
      llmCallback: LlmCallback,
      echo: Boolean,
      temperature: Float,
      numBos: Int,
      numEos: Int,
  ): Int

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param config the config for generation
   * @param llmCallback callback object to receive results
   */
  fun generate(prompt: String, config: LlmGenerationConfig, llmCallback: LlmCallback): Int =
      generate(
          null,
          0,
          0,
          0,
          prompt,
          config.seqLen,
          llmCallback,
          config.echo,
          config.temperature,
          config.numBos,
          config.numEos,
      )

  /**
   * Start generating tokens from the module.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results.
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  fun generate(
      image: IntArray?,
      width: Int,
      height: Int,
      channels: Int,
      prompt: String,
      seqLen: Int,
      llmCallback: LlmCallback,
      echo: Boolean,
  ): Int =
      generate(
          image,
          width,
          height,
          channels,
          prompt,
          seqLen,
          llmCallback,
          echo,
          DEFAULT_TEMPERATURE,
          DEFAULT_BOS,
          DEFAULT_EOS,
      )

  /**
   * Start generating tokens from the module.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results.
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   * @param temperature temperature for sampling (use negative value to use module default)
   */
  fun generate(
      image: IntArray?,
      width: Int,
      height: Int,
      channels: Int,
      prompt: String,
      seqLen: Int,
      llmCallback: LlmCallback,
      echo: Boolean,
      temperature: Float,
  ): Int =
      generate(
          image,
          width,
          height,
          channels,
          prompt,
          seqLen,
          llmCallback,
          echo,
          temperature,
          DEFAULT_BOS,
          DEFAULT_EOS,
      )

  /**
   * Start generating tokens from the module.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results.
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   * @param temperature temperature for sampling (use negative value to use module default)
   * @param numBos number of BOS tokens to prepend
   * @param numEos number of EOS tokens to append
   */
  fun generate(
      image: IntArray?,
      width: Int,
      height: Int,
      channels: Int,
      prompt: String,
      seqLen: Int,
      llmCallback: LlmCallback,
      echo: Boolean,
      temperature: Float,
      numBos: Int,
      numEos: Int,
  ): Int {
    if (image != null) {
      prefillImages(image, width, height, channels)
    }
    return generate(prompt, seqLen, llmCallback, echo, temperature, numBos, numEos)
  }

  // --- prefill methods ---

  /**
   * Prefill the KV cache with the given image input.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @return 0 on success
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  fun prefillImages(image: IntArray, width: Int, height: Int, channels: Int): Long {
    val nativeResult = prefillImagesInput(image, width, height, channels)
    if (nativeResult != 0) {
      throw RuntimeException("Prefill failed with error code: $nativeResult")
    }
    return 0
  }

  /**
   * Prefill a multimodal Module with the given image input via a direct ByteBuffer. The buffer data
   * is accessed directly without JNI array copies, unlike [prefillImages]. The ByteBuffer must
   * contain raw uint8 pixel data in CHW format with at least channels * height * width bytes
   * remaining. Only the first channels * height * width bytes from the buffer's current position
   * are read; the position of the original ByteBuffer is not modified.
   *
   * @param image Input image as a direct ByteBuffer containing uint8 pixel data
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @throws IllegalArgumentException if the ByteBuffer is not direct or has insufficient remaining
   *   bytes
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  fun prefillImages(image: ByteBuffer, width: Int, height: Int, channels: Int) {
    require(image.isDirect) { "Input ByteBuffer must be direct." }
    val expectedBytes: Long
    try {
      val pixels = Math.multiplyExact(width.toLong(), height.toLong())
      expectedBytes = Math.multiplyExact(pixels, channels.toLong())
    } catch (ex: ArithmeticException) {
      throw IllegalArgumentException(
          "width*height*channels is too large and overflows the allowed range.", ex)
    }
    require(
        width > 0 &&
            height > 0 &&
            channels > 0 &&
            expectedBytes <= Int.MAX_VALUE &&
            image.remaining() >= expectedBytes) {
          "ByteBuffer remaining (${image.remaining()}) must be at least width*height*channels ($expectedBytes)."
        }
    // slice() so that getDirectBufferAddress on the native side returns a pointer
    // starting at the current position, not the base address.
    val nativeResult = prefillImagesInputBuffer(image.slice(), width, height, channels)
    if (nativeResult != 0) {
      throw RuntimeException("Prefill failed with error code: $nativeResult")
    }
  }

  /**
   * Prefill a multimodal Module with the given normalized image input via a direct ByteBuffer. The
   * buffer data is accessed directly without JNI array copies, unlike [prefillImages]. The
   * ByteBuffer must contain normalized float pixel data in CHW format with at least channels *
   * height * width * 4 bytes remaining. Only the first channels * height * width floats from the
   * buffer's current position are consumed. The buffer must use the platform's native byte order
   * (set via `buffer.order(ByteOrder.nativeOrder())`).
   *
   * @param image Input normalized image as a direct ByteBuffer containing float pixel data in
   *   native byte order
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @throws IllegalArgumentException if the ByteBuffer is not direct, has insufficient remaining
   *   bytes, is not float-aligned, or does not use native byte order
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  fun prefillNormalizedImage(image: ByteBuffer, width: Int, height: Int, channels: Int) {
    require(image.isDirect) { "Input ByteBuffer must be direct." }
    require(image.order() == ByteOrder.nativeOrder()) {
      "Input ByteBuffer must use native byte order (ByteOrder.nativeOrder())."
    }
    require(image.position() % Float.SIZE_BYTES == 0) {
      "Input ByteBuffer position (${image.position()}) must be 4-byte aligned."
    }
    val expectedBytes: Long
    try {
      val wh = Math.multiplyExact(width, height)
      val whc = Math.multiplyExact(wh.toLong(), channels.toLong())
      val totalBytes = Math.multiplyExact(whc, Float.SIZE_BYTES.toLong())
      if (totalBytes > Int.MAX_VALUE) {
        throw IllegalArgumentException(
            "ByteBuffer size (width*height*channels*4) exceeds Integer.MAX_VALUE bytes: $totalBytes")
      }
      expectedBytes = totalBytes
    } catch (e: ArithmeticException) {
      throw IllegalArgumentException(
          "Overflow while computing width*height*channels*4 for ByteBuffer size.", e)
    }
    require(width > 0 && height > 0 && channels > 0 && image.remaining() >= expectedBytes) {
      "ByteBuffer remaining (${image.remaining()}) must be at least width*height*channels*4 ($expectedBytes)."
    }
    require(image.remaining() % Float.SIZE_BYTES == 0) {
      "ByteBuffer remaining (${image.remaining()}) must be a multiple of 4 (float size)."
    }
    // slice() so that getDirectBufferAddress on the native side returns a pointer
    // starting at the current position, not the base address.
    val nativeResult = prefillNormalizedImagesInputBuffer(image.slice(), width, height, channels)
    if (nativeResult != 0) {
      throw RuntimeException("Prefill failed with error code: $nativeResult")
    }
  }

  private external fun prefillImagesInput(
      image: IntArray,
      width: Int,
      height: Int,
      channels: Int,
  ): Int

  private external fun prefillImagesInputBuffer(
      image: ByteBuffer,
      width: Int,
      height: Int,
      channels: Int,
  ): Int

  private external fun prefillNormalizedImagesInputBuffer(
      image: ByteBuffer,
      width: Int,
      height: Int,
      channels: Int,
  ): Int

  /**
   * Prefill the KV cache with the given normalized image input.
   *
   * @param image Input normalized image as a float array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @return 0 on success
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  fun prefillImages(image: FloatArray, width: Int, height: Int, channels: Int): Long {
    val nativeResult = prefillNormalizedImagesInput(image, width, height, channels)
    if (nativeResult != 0) {
      throw RuntimeException("Prefill failed with error code: $nativeResult")
    }
    return 0
  }

  private external fun prefillNormalizedImagesInput(
      image: FloatArray,
      width: Int,
      height: Int,
      channels: Int,
  ): Int

  /**
   * Prefill the KV cache with the given preprocessed audio input.
   *
   * @param audio Input preprocessed audio as a byte array
   * @param batchSize Input batch size
   * @param nBins Input number of bins
   * @param nFrames Input number of frames
   * @return 0 on success
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  fun prefillAudio(audio: ByteArray, batchSize: Int, nBins: Int, nFrames: Int): Long {
    val nativeResult = prefillAudioInput(audio, batchSize, nBins, nFrames)
    if (nativeResult != 0) {
      throw RuntimeException("Prefill failed with error code: $nativeResult")
    }
    return 0
  }

  private external fun prefillAudioInput(
      audio: ByteArray,
      batchSize: Int,
      nBins: Int,
      nFrames: Int,
  ): Int

  /**
   * Prefill the KV cache with the given preprocessed audio input.
   *
   * @param audio Input preprocessed audio as a float array
   * @param batchSize Input batch size
   * @param nBins Input number of bins
   * @param nFrames Input number of frames
   * @return 0 on success
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  fun prefillAudio(audio: FloatArray, batchSize: Int, nBins: Int, nFrames: Int): Long {
    val nativeResult = prefillAudioInputFloat(audio, batchSize, nBins, nFrames)
    if (nativeResult != 0) {
      throw RuntimeException("Prefill failed with error code: $nativeResult")
    }
    return 0
  }

  private external fun prefillAudioInputFloat(
      audio: FloatArray,
      batchSize: Int,
      nBins: Int,
      nFrames: Int,
  ): Int

  /**
   * Prefill the KV cache with the given raw audio input.
   *
   * @param audio Input raw audio as a byte array
   * @param batchSize Input batch size
   * @param nChannels Input number of channels
   * @param nSamples Input number of samples
   * @return 0 on success
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  fun prefillRawAudio(audio: ByteArray, batchSize: Int, nChannels: Int, nSamples: Int): Long {
    val nativeResult = prefillRawAudioInput(audio, batchSize, nChannels, nSamples)
    if (nativeResult != 0) {
      throw RuntimeException("Prefill failed with error code: $nativeResult")
    }
    return 0
  }

  private external fun prefillRawAudioInput(
      audio: ByteArray,
      batchSize: Int,
      nChannels: Int,
      nSamples: Int,
  ): Int

  /**
   * Prefill the KV cache with the given text prompt.
   *
   * @param prompt The text prompt to prefill
   * @return 0 on success
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  fun prefillPrompt(prompt: String): Long {
    val nativeResult = prefillTextInput(prompt)
    if (nativeResult != 0) {
      throw RuntimeException("Prefill failed with error code: $nativeResult")
    }
    return 0
  }

  // returns status
  private external fun prefillTextInput(prompt: String): Int

  /**
   * Reset the context of the LLM. This will clear the KV cache and reset the state of the LLM. The
   * startPos will be reset to 0.
   */
  external fun resetContext()

  /** Stop current generate() before it finishes. */
  @DoNotStrip external fun stop()

  /** Force loading the module. Otherwise the model is loaded during first generate(). */
  @DoNotStrip external fun load(): Int

  companion object {
    const val MODEL_TYPE_TEXT = 1
    const val MODEL_TYPE_TEXT_VISION = 2
    const val MODEL_TYPE_MULTIMODAL = 2

    private const val DEFAULT_SEQ_LEN = 128
    private const val DEFAULT_ECHO = true
    private const val DEFAULT_TEMPERATURE = -1.0f
    private const val DEFAULT_BOS = 0
    private const val DEFAULT_EOS = 0
    private const val DEFAULT_LOAD_MODE = LlmModuleConfig.LOAD_MODE_MMAP

    @DoNotStrip
    @JvmStatic
    private external fun initHybrid(
        modelType: Int,
        modulePath: String,
        tokenizerPath: String,
        temperature: Float,
        dataFiles: List<String>,
        numBos: Int,
        numEos: Int,
        loadMode: Int,
    ): HybridData
  }
}
